#!/usr/bin/env python
# import paiv_utils as paiv
import argparse as ap
import json
import pandas as pd
import pathlib
import shutil

def copy_file_to_subdir(row,**kwargs) : #directory_in,directory_out
    # copy ofn/agm to new subdir ...
    #print(row)
    augm="na"
    if "augment_method" in row.keys() :
        augm = str(row["augment_method"])
    ofn= str(row["original_file_name"])
    fid = row.name[-8:len(row.name)]
    (ofn_root,ofn_extention) = row["original_file_name"].split('.')
    print("ofn : {}".format(ofn))
    print("ofn_root : {}".format(ofn_root))
    print("ofn_extention {}".format(ofn_extention))
    print("augm : {}".format(augm))
          
    newfile = "{}_{}_{}.{}".format(ofn_root,augm,fid,ofn_extention)
    print("newfile : {}".format(newfile))
    
    fin = kwargs["directory_in"]  + "/" + row.name + ".jpg"
    fout = kwargs["directory_out"] + "/" + row["category_name"] + "/"+ newfile
    print("fin : {}".format(fin))
    print("fout : {}".format(fout))
    print("=============")
    shutil.copy(fin,fout)


def reformat_paiv_cls_export(directory_in:str, directory_out:str="/tmp/output") :
    '''
    Function that will take an exported PAIV project that has classificatins, and re-organize
    the images into subdirectories
    directory_in : directory path of an unzipped exported dataset
    returns : 0 pass, 1 fail
    side effect : new subdirectories get written under directory_out using class name from prop.json
    '''
     
    with open(directory_in + '/prop.json') as json_file:
        data = json.load(json_file)
        df = pd.read_json(data['file_prop_info'], orient='records').set_index("_id")
    
    classes = list(df.category_name.unique())
    print("classes : {}".format(classes))
    # Make a directory in directory_out
    for c in classes :
        p = pathlib.Path(directory_out + "/" + c)
        print(str(p))
        if(not(p.exists())) :
           print("Creating a new sub directory {}".format(str(p)))
           p.mkdir(parents=True)

    # Now iterate thruogh each row of dataframe and COPY image to sudirectory
    df.apply(copy_file_to_subdir,directory_in=directory_in, directory_out=directory_out,axis=1)    


class SmartFormatterMixin(ap.HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser():
    parser = ap.ArgumentParser(description='Tool Reorganize an exported IBM Visual Insights Exported Directory'
                                           'Supports Classification data only and reorganizes it into folders'
                                           'with each subfolder being a label'
                                           'Example :'
                                           'python reorganize_exported_dataset.py --directory_in /tmp/exported_directory --directory_out /tmp/directory_out',
                               formatter_class=CustomFormatter)

    parser.add_argument(
        '--directory_in', action='store', nargs='?', required=True,
        help='S|--directory_in=<exported PAIV directory_path>'
             'Default: %(default)s')

    parser.add_argument(
        '--directory_out', action='store', nargs='?', required=True,
        help='S|--directory_out=<path where you want to resave data>'
             'Default: %(default)s')

    args = parser.parse_args()

    return args


def main():
    # Parse command line argument
    args = _parser()
    #args.force_refresh = True
    for argk in vars(args) :
        print(argk,vars(args)[argk])
    reformat_paiv_cls_export(args.directory_in, args.directory_out)

if __name__== "__main__":
  main()

# Todo : add threading for quicker video building
# Todo : add custom logic for tracking ball touches with denoising / smoothing
# Todo : add custom logic for displaying number of players at any given time with denoising / smoothing
