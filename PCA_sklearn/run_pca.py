#-------------------------------------------------------------------------------------------------#
#> @file   run_pca.py
#> @author Felippe M. Colombari
#> @brief  Very cool pca calculation with sklearn 
#> @date - Oct, 2023                                                           
#> - initial version and tests                                                
#> @date - XXX, 2024                                                           
#> - add class to write pseudotrajectories
#> @date - Jan 2025
#> - all calculation options via argparse; add cmd line verifications
#> - add options to check selection and read eigenvevtors to scale pseudotrajs without recalc everything
#> @todo 
#> - improve checks for integer and positive inputs
#> - check input files (including eigenvectors when needed)
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

#-------------------------------------------------------------------------------------------------#
#------------- functions used to check input files and options used ------------------------------#

def check_positive(numeric_type):
    def require_positive(value):
        number = numeric_type(value)
        if number <= 0:
            print(f"\tNumber {value} must be positive.")
            exit()
        return number
    return require_positive


def check_file(value):
    if os.path.isfile(value):
        print(f"\tFile '{value}' found\n")
    else:
        print(f"\tFile '{value}' not found\n")
        exit()

#-------------------------------------------------------------------------------------------------#
#------------- load trajectory and align it to the "align" atoms of the first frame --------------#
def Load_files():
    global traj, aligned_traj

    check_file( pdb_file )
    check_file( trj_file )

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

    def extract_template(self, pdb_file):
        file = open(pdb_file).readlines()
        for line in file:
            self.template.append(line[:31] + " " * 23 + line[54:])

    def load_coor(self, coordinates):
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
    
    def write_file(self, file_direction, n_model):
        file = open(file_direction, "w")
        startmodel = "MODEL " + str(n_model) + "\n"
        endmodel   = "ENDMDL\n"
        file.write(startmodel)
        for line in self.cur_cor:
            file.write(line)
        file.write(endmodel)
        file.close()

    def append_file(self, file_direction, n_model):
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
    pca1 = PCA( )

    reduced_cartesian = pca1.fit_transform( data )

    eigenvectors = pca1.components_

    #-------------------------------------------------------------------------------------------------#
    #------ use the pdb template created earlier and write a new pdb file with avg coordinates -------#
    protein = Protein()
    protein.extract_template( "template_selection.pdb" )

    # multiply by 10 to convert from nm (xtc) to angstrom (pdb)
    avg_angstrom = compute_average_structure( traj_pca.xyz ) * 10.000

    protein.load_coor( avg_angstrom.reshape(-1,1) )
    protein.write_file( "average_structure.pdb", 1 )

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

    df_eigenval_cumsum = pd.DataFrame( np.cumsum( pca1.explained_variance_ratio_) , columns = [ "#n var" ] )
    df_eigenval_cumsum.to_csv( r'eigenvalues_cumsum.dat', index = True, sep = '\t', float_format = "%12.6f" )

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
    global eigenvectors

    out_vectors = "eigenvectors.dat"

    try:
        os.remove( out_vectors )
    except OSError:
        pass

    for pca in range(0,total_pc_output):
        str_header = [ "--Eingenvector_" + str(pca), "  X,Y,Z ", "coordinates--"]
        df_eigenvectors = pd.DataFrame( eigenvectors[pca].reshape( template.n_atoms, 3 ))
        df_eigenvectors.to_csv(out_vectors, header=str_header, index=None, sep='\t', float_format='%12.8f', mode = 'a')
    
def Read_eigenvectors( total_pc_output ):
    global eigenvectors, avg_angstrom

    check_file( "average_structure.pdb" )

    template     = md.load( "average_structure.pdb", top = "average_structure.pdb" )
    avg_angstrom = template.xyz * 10.0
    
    eigenvectors = [0] * total_pc_output 
    in_vectors   = "eigenvectors.dat"

    check_file( in_vectors )
    
    with open(in_vectors) as vectors_file:
        testsite_array = vectors_file.readlines()

    for pca in range(0,total_pc_output):
        for n in range(0,template.n_atoms):
            first = 1 + (template.n_atoms + 1) * pca
            last  = (template.n_atoms + 1) * (pca + 1)

        eigenvectors[pca] = np.array(' '.join(testsite_array[first:last]).split(), dtype=float)

#-------------------------------------------------------------------------------------------------#
#------------------------ write eingenvectors on pseudotrajectory files --------------------------#
def Write_pseudotrajs():
    global coords, filename, model_nr

    protein = Protein()
    protein.extract_template( "template_selection.pdb" )

    # pseudotraj_steps to negative displ. + avg structure + pseudotraj_steps to positive displ.
    total_frames = 2 * pseudotraj_steps + 1 

    for pca in range(0,total_pc_output):
        print( '\n\tWriting eigenvector %s to file... ' % pca , end = '' )
    
        filename = "trj_eigenvector_" + str( pca ) + ".pdb"

        try:
            os.remove( filename )
        except OSError:
            pass

        for steps in range(-pseudotraj_steps,pseudotraj_steps):
            coords = avg_angstrom.reshape( -1, 1 ) - eigenvectors[pca].reshape( -1, 1 ) * steps * pseudotraj_scalf
            protein.load_coor( coords )
            model_nr = steps + pseudotraj_steps + 1
            protein.append_file( filename, model_nr )

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

    parser_a = subparsers.add_parser('check', help='Requires: -p/--pdb, -t/--trj, -f/--fit, -s/--sel')
    parser_a.add_argument('-p', '--pdb', required=True, help = pdb_help)
    parser_a.add_argument('-t', '--trj', required=True, help = trj_help)
    parser_a.add_argument('-f', '--fit', required=True, help = fit_help)
    parser_a.add_argument('-s', '--sel', required=True, help = sel_help)

    parser_b = subparsers.add_parser('run', help='Requires: -p/--pdb, -t/--trj, -f/--fit, -s/--sel, -d/--dsp, -n/--num')
    parser_b.add_argument('-p', '--pdb', required=True, help = pdb_help)
    parser_b.add_argument('-t', '--trj', required=True, help = trj_help)
    parser_b.add_argument('-f', '--fit', required=True, help = fit_help)
    parser_b.add_argument('-s', '--sel', required=True, help = sel_help)
    parser_b.add_argument('-d', '--dsp', required=True, help = dsp_help, type=check_positive(float))
    parser_b.add_argument('-n', '--num', required=True, help = num_help, type=check_positive(int))

    parser_c = subparsers.add_parser('rescale', help='Requires: -d/--dsp, -n/--num')
    parser_c.add_argument('-d', '--dsp', required=True, help = dsp_help)
    parser_c.add_argument('-n', '--num', required=True, help = num_help)

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
