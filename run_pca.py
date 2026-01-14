#-------------------------------------------------------------------------------------------------#
#> @file   run_pca.py
#> @author Felippe M. Colombari
#> @brief  Very cool pca calculation with sklearn 
#> @date - Oct, 2023                                                           
#> - initial version and tests                                                
#> @date - Jun 2024                                                           
#> - add class to write pseudotrajectories
#> @date - Jan 2025
#> - all calculation options via argparse; add cmd line verifications
#> - write eigenvectors to file
#> - added option to check selection 
#> - added option to read eigenvectors and scale pseudotrajs without full calculation
#> @date - Mar 2025
#> - modify code to calculate eigenvectors norm and write them to each pseudotraj bfactor field
#> @date - Jun 2025
#> - write covariance matrix to files (ascii and image)
#> @todo 
#> - improve checks for integer and positive inputs
#> - improve memory usage and parallelization
#---------------------------------------------------------------------------------------------------

import os
import sys
import argparse
import numpy as np
import pandas as pd
import mdtraj as md
from mdtraj.geometry.alignment import compute_average_structure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter

#-------------------------------------------------------------------------------------------------#
#------------- functions used to check input files and options used ------------------------------#

def Check_positive(numeric_type):
    def Require_positive(value):
        number = numeric_type(value)
        if number <= 0:
            print(f"\tNumber {value} must be positive.")
            print('|' + '=' * term_size + '|\n')
            exit()
        return number
    return Require_positive

def Check_file(value):
    if os.path.isfile(value):
        print(f"\tFile '{value}' found\n")
    else:
        print(f"\tFile '{value}' not found. ERROR!\n")
        print('|' + '=' * term_size + '|\n')
        exit()

#-------------------------------------------------------------------------------------------------#
#------------- load trajectory and align it to the "align" atoms of the first frame --------------#
def Load_files():
    global traj, aligned_traj

    Check_file( pdb_file )
    
    Check_file( trj_file )

    traj = md.load( trj_file, top = pdb_file )

    align_selection = traj.top.select( sel_fit )

    aligned_traj = traj.superpose( traj[0], atom_indices = align_selection )

#-------------------------------------------------------------------------------------------------#
#------------- save pdb template containing only the atoms used for analysis ---------------------#
def Save_template():
    global pca_selection, template

    pca_selection = aligned_traj.top.select( sel_pca )

    template = aligned_traj.atom_slice( pca_selection )[0]

    template.save_pdb( "template_selection_full.pdb" )

    with open( "template_selection_full.pdb", "r" ) as inputFile, open( "template_selection.pdb", "w" ) as outFile:
        for line in inputFile:
            if line.startswith( "ATOM" ) or line.startswith( "HETATM" ):
                outFile.write(line)

#-------------------------------------------------------------------------------------------------#
#-------- this class contains functions needed to write pdb coordinates on a pdb template --------#
class Protein():
    def __init__(self):
        self.template = []
        self.cur_cor = None

    def Extract_template(self, pdb_file):
        file = open(pdb_file).readlines()
        for line in file:
            self.template.append(line[:31] + " " * 23 + line[54:])

    def Load_coor(self, coordinates):
        self.cur_cor = self.template[:]
        index = 0
        for i in range(len(coordinates) // 3):
            x, y, z = coordinates[3 * i: 3 * (i + 1)]
            #print(x,y,z)

            x, y, z = "%.3f" % x[0], "%.3f" % y[0], "%.3f" % z[0]
            #print("new",x,y,z)

            for cor, end in zip ([x, y, z], [38, 46, 54]):
                length = len(cor)
                self.cur_cor[index] = self.cur_cor[index][:end -length] \
                        + cor + self.cur_cor[index][end:]
            index += 1
   
    def Set_bfactors(self, bfactors):
        if len(bfactors) != len(self.cur_cor):
            raise ValueError("Number of B-factors must match number of atoms")

        for i in range(len(self.cur_cor)):
            if self.cur_cor[i].startswith(("ATOM  ", "HETATM")):
                bfactor_str = "%6.2f" % float(bfactors[i])
                self.cur_cor[i] = self.cur_cor[i][:60] + bfactor_str + self.cur_cor[i][66:]

    def Write_file(self, file_direction, n_model):
        file = open(file_direction, "w")
        startmodel = "MODEL " + str(n_model) + "\n"
        endmodel   = "ENDMDL\n"
        file.write(startmodel)

        for line in self.cur_cor:
            file.write(line)

        file.write(endmodel)
        file.close()

    def Append_file(self, file_direction, n_model):
        file = open(file_direction, "a")
        startmodel = "MODEL " + str(n_model) + "\n"
        endmodel   = "ENDMDL\n"
        file.write(startmodel)
        for line in self.cur_cor:
            file.write(line)
        file.write(endmodel)
        file.close()

def Calculate_pca():
    global pca1, avg_angstrom, eigenvectors
    
    #-------------------------------------------------------------------------------------------------#
    #------------- define new trajectory with the appropriate atom selection -------------------------#
    traj_pca = aligned_traj.atom_slice( pca_selection )

    data = traj_pca.xyz[:].reshape( traj.n_frames, -1 )

    #-------------------------------------------------------------------------------------------------#
    #------------------- fit the model with X and apply the dim. reduction on X ----------------------#
    pca1 = PCA()

    reduced_cartesian = pca1.fit_transform( data )

    eigenvectors = pca1.components_

    #-------------------------------------------------------------------------------------------------#
    #------ use the pdb template created earlier and write a new pdb file with avg coordinates -------#
    protein = Protein()
    
    protein.Extract_template( "template_selection.pdb" )

    # multiply by 10 to convert from nm (xtc) to angstrom (pdb)
    avg_angstrom = compute_average_structure( traj_pca.xyz ) * 10.000

    protein.Load_coor( avg_angstrom.reshape(-1,1) )
    
    protein.Write_file( "average_structure.pdb", 1 )

    #-------------------------------------------------------------------------------------------------#
    #------------------------------- write pca eigenvalues to file -----------------------------------#
    print( '\n\tWriting eigenvalues to file... ', end = '' )

    df_eigenval = pd.DataFrame( pca1.explained_variance_, columns = [ "#n var" ] )
    
    df_eigenval.to_csv( r'eigenvalues.dat', index = True, sep = '\t', float_format = "%12.6f" )

    print( 'DONE')

    #-------------------------------------------------------------------------------------------------#
    #--------------------------- write normalized pca eigenvalues to file ----------------------------#
    print( '\n\tWriting eigenvalues ratio to file... ', end = '' )

    df_eigenval_ratio = pd.DataFrame( pca1.explained_variance_ratio_, columns = [ "#n var" ] )
    
    df_eigenval_ratio.to_csv( r'eigenvalues_ratio.dat', index = True, sep = '\t', float_format = "%12.6f" )

    print( 'DONE')

    #-------------------------------------------------------------------------------------------------#
    #------------------------ write cumulative sum of pca eigenvalues to file ------------------------#
    print( '\n\tWriting eigenvalues cumulative sum to file... ', end = '' )

    df_eigenval_csum = pd.DataFrame( np.cumsum( pca1.explained_variance_ratio_) , columns = [ "#n var" ] )
    
    df_eigenval_csum.to_csv( r'eigenvalues_csum.dat', index = True, sep = '\t', float_format = "%12.6f" )

    print( 'DONE')

    #-------------------------------------------------------------------------------------------------#
    #--------------------------------- write pcs along time to file ----------------------------------#
    print( '\n\tWriting pcs along time to file... ', end = '' )

    column_names = [ f"#pc{i:05d}" for i in range( 1, total_pc_output + 1 ) ]

    df = pd.DataFrame( reduced_cartesian[:,0:total_pc_output], columns = [ column_names ] ).reset_index(names="#t")

    df.to_csv( r'all_pcs.dat', sep = '\t', index=False, float_format = "%9.6f" )

    print( 'DONE')

#-------------------------------------------------------------------------------------------------#
#------------------------------ write eingenvectors to output files ------------------------------#
def Write_eigenvectors():
    global eigenvectors, df_eigenvectors_norm

    out_vectors = "eigenvectors.dat"
    out_norms   = "eigenvectors_norm.dat"
    df_eigenvectors_norm = [0] * total_pc_output
    all_norms = []

    try:
        os.remove( out_vectors )
        os.remove( out_norms )
    except OSError:
        pass

    for pca in range(0,total_pc_output):
        df_eigenvectors = pd.DataFrame( eigenvectors[pca].reshape( template.n_atoms, 3 ))
        df_eigenvectors_norm[pca] = pd.DataFrame( np.linalg.norm(eigenvectors[pca].reshape( template.n_atoms, 3 ), axis=1).reshape( template.n_atoms,1) )[0]
        df_eigenvectors_merged = pd.concat([df_eigenvectors, df_eigenvectors_norm[pca]], axis=1)

        #print(df_eigenvectors_merged)
        label_vx = f" v_{pca}_x".center(12)
        label_vy = f" v_{pca}_y".center(12)
        label_vz = f" v_{pca}_z".center(12)
        label_vnorm = f" || v_{pca} ||".center(12)
        str_header = [ label_vx, label_vy, label_vz, label_vnorm]
        
        df_eigenvectors_merged.to_csv(out_vectors, header=str_header, index=None, sep='\t', float_format='%12.9f', mode = 'a')

def Read_eigenvectors( total_pc_output ):
    global eigenvectors, avg_angstrom, df_eigenvectors_norm

    Check_file( "average_structure.pdb" )

    template     = md.load( "average_structure.pdb", top = "average_structure.pdb" )
    avg_angstrom = template.xyz * 10.0
   
    tmp                  = []
    eigenvectors         = [0] * total_pc_output 
    df_eigenvectors_norm = [0] * total_pc_output 
    in_vectors           = "eigenvectors.dat"

    check_file( in_vectors )
    
    with open(in_vectors) as vectors_file:
        testsite_array = vectors_file.readlines()

    for pca in range(0,total_pc_output):
        for n in range(0,template.n_atoms):
            first = 1 + (template.n_atoms + 1) * pca
            last  = (template.n_atoms + 1) * (pca + 1)
        tmp = np.array(' '.join(testsite_array[first:last]).split(), dtype=float)

        mask1 = np.ones(len(tmp), dtype=bool)
        mask1[3::4] = False  # just set every 4th element to False
        filtered_arr = tmp[mask1].reshape(-1,3)
        eigenvectors[pca] = filtered_arr
        
        mask2 = np.zeros(len(tmp), dtype=bool)
        mask2[3::4] = True  # or set every 4th element to True
        filtered_arr = tmp[mask2].reshape(-1,1)
        df_eigenvectors_norm[pca] = filtered_arr

#-------------------------------------------------------------------------------------------------#
#------------------------ write eingenvectors on pseudotrajectory files --------------------------#
def Write_pseudotrajs():
    global coords, filename, model_nr

    protein = Protein()
    protein.Extract_template( "template_selection.pdb" )

    # pseudotraj_steps to negative displ. + avg structure + pseudotraj_steps to positive displ.
    total_frames = 2 * pseudotraj_steps + 1 

    for pc in range(0,total_pc_output):
        print( '\n\tWriting eigenvector %s to file... ' % pc , end = '' )
    
        filename = "trj_eigenvector_" + str( pc ) + ".pdb"

        try:
            os.remove( filename )
        except OSError:
            pass

        for steps in range( -pseudotraj_steps, pseudotraj_steps+1 ):
            coords = avg_angstrom.reshape( -1, 1 ) + eigenvectors[pc].reshape( -1, 1 ) * steps * pseudotraj_scalf
            protein.Load_coor( coords )
            model_nr = steps + pseudotraj_steps + 1
            bfactors = df_eigenvectors_norm[pc]
            protein.Set_bfactors( bfactors )
            protein.Append_file( filename, model_nr )

        print( 'DONE' )

#-------------------------------------------------------------------------------------------------#
#------------------------ write covariance matrix to ASCII and PNG files -------------------------#
def Get_covar_matrix():

    print( '\n\tWriting covariance matrix files... ', end = '' )
    
    cov_matrix = pca1.get_covariance()
    np.savetxt("covar_matrix.dat", cov_matrix, fmt="%.6f", delimiter=',')
    
    max_x   = len(cov_matrix[0])
    max_y   = len(cov_matrix[1])
    min_cov = np.min(cov_matrix)
    max_cov = np.max(cov_matrix)

    colors = [ ( min_cov, "#3333CC" ), ( 0.0, "#EEEEEE" ), ( max_cov, "#CC3333" ) ]
    cmap   = LinearSegmentedColormap.from_list( "custom", [ c for v,c in colors ] )
    norm   = TwoSlopeNorm( vmin = min_cov, vcenter = 0.0, vmax = max_cov )
    cbar   = { "ticks": [ min_cov, 0.0, max_cov ], "format": StrMethodFormatter("{x:.1E}") } 
    
    ax = sns.heatmap( cov_matrix, clip_on = False, norm = norm, cmap = cmap, cbar_kws = cbar ) 
   
    for _, spine in ax.spines.items():
        spine.set_visible(True)  
        spine.set_linewidth(1)   
        spine.set_color('black') 

    min_yticks = np.arange( 0.5, max_y + 0.5, 1 )
    maj_yticks = np.arange( 0, max_y + 1, 5 )    
    ytick_pos  = maj_yticks + 0.5

    ax.invert_yaxis()
    ax.set_yticks( ytick_pos )
    ax.set_yticks( min_yticks, minor = True )
    ax.set_yticklabels( maj_yticks, rotation = 0 ) 

    min_xticks = np.arange( 0.5, max_x + 0.5, 1 )  
    maj_xticks = np.arange( 0, max_x + 1, 5 ) 
    xtick_pos  = maj_xticks + 0.5
    
    ax.set_xticks( xtick_pos )
    ax.set_xticks( min_yticks, minor = True )
    ax.set_xticklabels( maj_xticks, rotation = 0 ) 

    ax.set_ylabel("Atom index")
    ax.set_xlabel("Atom index")
    plt.title("Covariance matrix")

    plt.savefig("covar_matrix.png", format="png", bbox_inches="tight", dpi=300)
    plt.close()

    print( 'DONE' )
    
    print( '\n\tWriting RMSF file... ', end = '' )
    
    diag = np.diag(cov_matrix.astype(np.float32))

    rmsf = np.sqrt(diag.reshape(-1,3).sum(axis=1))
    np.savetxt("rmsf.dat", rmsf, fmt="%.6f", delimiter=',')
    
    print( 'DONE' )

######################################### Command line arguments parsing ##############################################

description = "Scikit-learn implementation of PCA analysis for MD trajectories"

def Read_cmdline():
    global args

    parser = argparse.ArgumentParser( description = description )
    subparsers = parser.add_subparsers( dest = 'command', required = True, help = 'Available commands. Run "run_pca.py command -h" for additional help' )

    pdb_help = "PDB file used as molecular topology"
    trj_help = "Trajectory file"
    fit_help = "Atom selection to perform trj fitting"
    sel_help = "Atom selection to perform pca calculation"
    dsp_help = "Scaling factor for pseudotrajectories displacements"
    num_help = "Number of principal components to analyze"

    parser_a = subparsers.add_parser( 'check', help = 'Requires: -p/--pdb, -t/--trj, -f/--fit, -s/--sel' )
    parser_a.add_argument( '-p', '--pdb', required = True, help = pdb_help ) 
    parser_a.add_argument( '-t', '--trj', required = True, help = trj_help )
    parser_a.add_argument( '-f', '--fit', required = True, help = fit_help )
    parser_a.add_argument( '-s', '--sel', required = True, help = sel_help )

    parser_b = subparsers.add_parser( 'run', help = 'Requires: -p/--pdb, -t/--trj, -f/--fit, -s/--sel, -d/--dsp, -n/--num' )
    parser_b.add_argument( '-p', '--pdb', required = True, help = pdb_help )
    parser_b.add_argument( '-t', '--trj', required = True, help = trj_help )
    parser_b.add_argument( '-f', '--fit', required = True, help = fit_help )
    parser_b.add_argument( '-s', '--sel', required = True, help = sel_help )
    parser_b.add_argument( '-d', '--dsp', required = True, help = dsp_help, type = Check_positive(float) )
    parser_b.add_argument( '-n', '--num', required = True, help = num_help, type = Check_positive(int) )

    parser_c = subparsers.add_parser( 'rescale', help = 'Requires: -d/--dsp, -n/--num' )
    parser_c.add_argument( '-d', '--dsp', required = True, help = dsp_help )
    parser_c.add_argument( '-n', '--num', required = True, help = num_help )

    args = parser.parse_args()

#-------------------------------------------------------------------------------------------------#
#------------------------------------------ Run script -------------------------------------------#
if __name__ == '__main__':
    Read_cmdline()

    term_size = 105
    print('\n|' + '=' * term_size + '|\n')

    if args.command == "check":
        print("\tJust writing selected atoms to file...\n")
        pdb_file          = args.pdb
        trj_file          = args.trj
        sel_fit           = args.fit
        sel_pca           = args.sel
        Load_files()
        Save_template()
    elif args.command == "run":
        print("\tStarting full PCA calculations...\n")
        pdb_file          = args.pdb
        trj_file          = args.trj
        sel_fit           = args.fit
        sel_pca           = args.sel
        total_pc_output   = int(args.num)
        pseudotraj_scalf  = float(args.dsp)
        pseudotraj_steps  = 15
        Load_files()
        Save_template()
        Calculate_pca()
        Write_eigenvectors()
        Write_pseudotrajs()
        Get_covar_matrix()
    elif args.command == "rescale":
        print("\tStarting pseudotrajectories generation...\n")
        total_pc_output   = int(args.num)
        pseudotraj_scalf  = float(args.dsp)
        pseudotraj_steps  = 15
        Read_eigenvectors(total_pc_output)
        Write_pseudotrajs()
    else:
        print("\tInvalid runtype option. Must be \"check\", \"run\" or \"rescale\".\n")

    print('\n|' + '=' * term_size + '|\n')
