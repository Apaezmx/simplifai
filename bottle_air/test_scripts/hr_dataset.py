# -*- coding: utf-8 -*-
"""
Script to get train.csv and test.csv by splitting a dataset.
"""

from sklearn.model_selection import train_test_split

"""
Path where your dataset is saved.
"""
PATH_FILE = 'HR.csv'

"""
Boolean to tell the script if you want to change an int column/s to str column/s.
"""
CHANGE_INT_TO_STR = True
"""
If CHANGE_INT_TO_STR is True, then write a list with the indeces of the columns that you want
to change.
"""
IDX_TO_CHANGE = [2]
"""
If CHANGE_INT_TO_STR is True, then write the letter that you want to add to each item of the
int column/s that you want to change.
"""
LETTER_TO_ADD = 'p'

"""
Choose the size of your test data (1 == the whole data).
"""
TEST_SIZE = 0.1
"""
Choose whether to shuffle the dataset before splitting it or not.
"""
SHUFFLE_DATA = True

"""
Path to save the train data.
"""
PATH_TRAIN = 'train.csv'
"""
Path to save the test data.
"""
PATH_TEST = 'test.csv'

"""
Functions
"""

def read_csv(path_file):
    """
    Read the dataset and return a list containing all the rows in the table and the first row
    in the dataset (the one that contains the names of the columns).
    Check the ending of your rows. If they finish with '\n' only, change line[:-2] to line[:-1];
    if they end with '\r\n', leave it as it is.
    =Parameters=
    path_file = Path where your dataset is saved.
    """
    
    data_list = []
    with open(path_file, 'r') as f:
        for line in f.readlines():
            data_list.append(line[:-2].split(','))
    first_row = data_list.pop(0)
    
    return data_list, first_row


def change_int_to_str(lst, idx_to_change, letter_to_add):
    """
    Change a column of ints to a column of strings by adding a letter. This is used to 'fool' the
    neural network into thinking that a number is a string.
    =Parameters=
    lst: The list containing your dataset.
    idx_to_change: List containing the indices of the columns that you want to transform.
    letter_to_add: String with the letter that you want to insert into each item.
    """
    
    new_lst = []
    for item in lst:
        for idx in idx_to_change:
            item[idx] = letter_to_add + item[idx]
        new_lst.append(item)
        
    return new_lst
    

def split_the_dataset(lst, first_row, test_size, shuffle_data):
    """
    Split the dataset into two lists: train data and test data.
    =Parameters=
    lst: List containing your dataset.
    first_row: The first row of your dataset containing the names of the columns.
    test_size: The size of your test data.
    shuffle: Whether you want to shuffle the data before splitting or not.
    """

    train, test = train_test_split(lst, test_size=test_size, shuffle=shuffle_data)
    train.insert(0, first_row)
    test.insert(0, first_row)
    
    return train, test


def save_csv(train_data, test_data, train_path, test_path):
    """
    Saves the train data and the test data in train.csv and test.csv.
    =Parameters=
    train_path: Path where you want to save train.csv.
    test_path: Path where you want to save test.csv.
    """
    
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(','.join(item)+'\n')
    
    with open(test_path, 'w') as f:
        for item in test_data:
            f.write(','.join(item)+'\n')
            
            
def run_split(path_file, idx_to_change, letter_to_add, test_size, shuffle_data, 
              train_path, test_path):
    """
    Run the script.
    =Parameters=
    path_file = Path where your dataset is saved.
    idx_to_change: List containing the indices of the columns that you want to transform.
    letter_to_add: String with the letter that you want to insert into each item.
    test_size: The size of your test data.
    shuffle_data: Whether you want to shuffle the data before splitting or not.
    train_path: Path where you want to save train.csv.
    test_path: Path where you want to save test.csv.
    """
    
    data_list, first_row = read_csv(path_file)
    if CHANGE_INT_TO_STR:
        changed_data = change_int_to_str(data_list, idx_to_change, letter_to_add)
        train, test = split_the_dataset(changed_data, first_row, test_size, shuffle_data)
        save_csv(train, test, train_path, test_path)
    else:
        train, test = split_the_dataset(data_list, first_row, test_size, shuffle_data)
        save_csv(train, test, train_path, test_path)
        

if __name__ == "__main__":
    run_split(PATH_FILE, IDX_TO_CHANGE, LETTER_TO_ADD, TEST_SIZE, SHUFFLE_DATA,
              PATH_TRAIN, PATH_TEST)





        
