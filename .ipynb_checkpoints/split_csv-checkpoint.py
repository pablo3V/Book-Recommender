####################################################################################################
#                                                                                                  #
# Python program to split large .csv files in smaller chunks                                       #
#                                                                                                  #
# This code is a slight modification of the original code in:                                      # 
# https://stackoverflow.com/questions/20033861/how-can-i-split-a-large-file-csv-file-7gb-in-python #
#                                                                                                  #
# Note: added comments and arguments to the main() funcion                                         #
#                                                                                                  #
####################################################################################################

# Import libraries
import argparse # library to handle inputs in command line in Python


def main(file_name, n_rows, n_chunks):

    # file_name : name of the file to split
    # n_rows : number of rows of the csv file
    # n_chunks : number of chunks

    chunk_size = int(int(n_rows) / (int(n_chunks) - 1)) # number of lines of each part

    def write_chunk(part, lines):
        with open(str(file_name) + '_part_'+ str(part) +'.csv', 'w') as f_out:
            f_out.write(header)
            f_out.writelines(lines)

    with open(str(file_name) + '.csv', 'r') as f:
        count = 0
        header = f.readline() # header of the table in the file
        lines = []
        
        for line in f:
            count += 1
            lines.append(line)
            if count % chunk_size == 0:
                write_chunk(count // chunk_size, lines)
                lines = [] 
                               
        # if n_rows / n_chunks is not an integer, there will be lines remaining
        # write the remainder
        if len(lines) > 0:
            write_chunk((count // chunk_size) + 1, lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a large CSV file into smaller chunks.')    
    parser.add_argument('input_file', help='The name (without the .csv) of the input CSV file to be split')
    parser.add_argument('n_rows', help='The number of rows of the .csv file to be split')
    parser.add_argument('n_chunks', help='The number of chunks for the .csv file to be split')
    args = parser.parse_args()
    
    # in the terminal, use: python split_csv.py file_name n_rows n_chunks
    main(args.input_file, args.n_rows, args.n_chunks)
    
