# -*- coding: utf-8 -*-
"""
Testing script for infer.html
"""

from selenium import webdriver
import selenium.webdriver.chrome.service as service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

"""
Path where the Selenium driver for your browser is saved.
Go to http://selenium-python.readthedocs.io/installation.html to search for the
appropiate driver and then copy it to the directory where this script is saved.
"""
PATH_TO_DRIVER = './chromedriver'
"""
Path where your web browser application is saved.
"""
PATH_TO_BROWSER = 'menu://applications/Internet/google-chrome.desktop'
"""
The url where infer.html is found.
"""
INFER_URL = 'http://localhost:8012/infer.html'
"""
Name of the model that you want to use to make inferences.
"""
MODEL = 'e07c99276f'
"""
Path to the test csv file.
"""
PATH_TO_TEST = 'hr_test.csv'
"""
Name of the column that you want to predict and its type (int, float, or str).
"""
KEY_TO_INFER = ['output_left', int]
"""
Names of the columns that are integers.
"""
COLUMNS_INT = ['average_montly_hours', 'time_spend_company', 'Work_accident',
               'promotion_last_5years']
"""
Names of columns that are floats.
"""
COLUMNS_FLOAT = ['satisfaction_level', 'last_evaluation']
"""
Names of the columns that are strings.
"""
COLUMNS_STR = ['number_project', 'sales', 'salary']
"""
Path to save targets.
"""
PATH_TARGETS = 'targets.txt'
"""
Path to save predictions.
"""
PATH_PRED = 'predictions.txt'
"""
Path to save accuracy.
"""
PATH_ACC = 'accuracy.txt'

"""
Functions
"""

def get_column_names(path_to_test, key_to_infer):
    """
    Get a list containing the names of the columns in your table minus the column that
    you want to infer.
    =Parameters=
    path_to_test: Path where your test.csv is saved.
    key_to_infer: The name of the column that you want to infer.
    """
    
    with open(path_to_test, 'r') as f:
        list_of_columns = f.readline()[:-1].split(',')
        list_of_columns.remove(key_to_infer[0])
    return list_of_columns


def start_server_and_driver(path_to_driver, path_to_browser):
    """
    Start the Selenium server and driver and return them as objects.
    =Parameters=
    path_to_driver: The path where the Selenium driver for your browser is saved.
    path_to_browser: The path where your browser application is saved.
    """
    
    server = service.Service(path_to_driver)
    server.start()

    capabilities = {'chrome.binary': path_to_browser}
    driver = webdriver.Remote(server.service_url, capabilities)
    
    return server, driver
    
    
def stop_server_and_driver(server, driver):
    """
    Close the driver and then stop the server.
    =Parameters=
    driver: driver object returned by def start_server_and_driver()
    server: server object returned by def start_server_and_driver()
    """
    
    driver.close()
    server.stop()
    

def prepare_data(path_to_test, path_to_targets, key_to_infer, columns_int, columns_float):
    """
    Prepare the data for the tests. Returns a list with the names of the columns in your table
    and a list with dictionaries. Each dictionary represents a row in your table (key=column name;
    value=value for that row). It also saves the targets in a .txt for later use.
    =Parameters=
    path_to_test: The path where your test.csv is saved.
    path_to_targets: The path where you want to save targets.txt.
    key_to_infer: The column that you want to infer.
    columns_int: A list containing the names of the columns that are integers.
    columns_float: A list containing the names of the columns that are floats.
    """
     
    column_names = get_column_names(path_to_test, key_to_infer)
    
    with open(path_to_test, 'r') as f:
        test_dict_unord = csv.DictReader(f)
        test_dict_ord = []
        targets = []
        for row in test_dict_unord:
            new_dict = {}
            for key, value in row.items():
                if key in columns_int:
                    new_dict[key] = int(value)
                elif key in columns_float:
                    new_dict[key] = float(value)
                else:
                    new_dict[key] = str(value)
            test_dict_ord.append(new_dict)
            targets.append(key_to_infer[1](row[key_to_infer[0]]))
            
    with open(path_to_targets, 'w') as f:
        for target in targets:
            f.write(str(target)+'\n')
    
    return column_names, test_dict_ord
    
    
def do_inferences(driver, infer_url, path_to_preds, column_names, test_dict_ord):       
    """
    Do the tests. This uses Selenium to get predictions for each row in your table. It appends
    the predictions to predictions.txt. If you want to change the rate at which the predictions
    are appended, change the variable save_each.
    =Parameters=
    driver: Driver object returned by def start_server_and_driver()
    infer_url: Url where infer.html exists.
    path_to_preds: The path where you want to save predictions.txt.
    test_dict_ord: List of dictionaries representing each row in your table.
    """

    save_each = 100
    len_of_list = len(test_dict_ord)
    counter = 0
    
    driver.get(infer_url)
                
    predictions = []
    time1 = time.time()
    for row in test_dict_ord:
        
        text_model = WebDriverWait(driver, 10).until(\
            EC.presence_of_element_located((By.ID, 'text_model')))

        text_model.clear()
        text_model.send_keys(MODEL)

        infer_types = driver.find_element_by_id('btn_infer_types')
        infer_types.click()
        
        for name in column_names:
            _input = WebDriverWait(driver, 10).until(\
                EC.presence_of_element_located((By.NAME, name)))
            _input.clear()
            _input.send_keys(str(row[name]))
        
        btn_infer = driver.find_element_by_id('btn_infer')
        btn_infer.click()
        
        prediction_box = WebDriverWait(driver, 10).until(\
            EC.visibility_of_element_located((By.ID, 'new-prediction')))
        prediction = prediction_box.text
        predictions.append(prediction)
        driver.refresh()
        
        counter+=1
        if counter % save_each == 0:
            time2 = time.time()
            batch_time = time2 - time1
            print("Progress: {}/{}".format(counter, len_of_list))
            print("Time elapsed: {}/{}".format(str((batch_time)*(counter//save_each)), 
                  str(batch_time)*(len_of_list//save_each)))
            time1 = time2
            with open(path_to_preds, 'a') as f:
                for pred in predictions:
                    f.write(str(pred)+'\n')
            predictions = []
            
    with open(path_to_preds, 'a') as f:
                for pred in predictions:
                    f.write(str(pred)+'\n')
            
            
def run_test(path_to_driver, path_to_browser, path_to_test, path_targets, key_to_infer,
             columns_int, columns_float, infer_url, path_pred):
    """
    Runs the tests. At the end, you have a targets.txt and a predicions.txt to get the
    accuracy of your tests.
    =Parameters=
    path_to_driver: The path where the Selenium driver for your browser is saved.
    path_to_browser: The path where your browser application is saved.
    path_to_test: Path where your test.csv is saved.
    path_targets: The path where you want to save targets.txt.
    key_to_infer: The column that you want to infer.
    columns_int: A list containing the names of the columns that are integers.
    columns_float: A list containing the names of the columns that are floats.
    infer_url: Url where infer.html exists.
    path_pred: The path where you want to save predictions.txt.
    """
    
    server, driver = start_server_and_driver(path_to_driver, path_to_browser)
    column_names, test_dict_ord = prepare_data(path_to_test, path_targets, key_to_infer, 
                                                    columns_int, columns_float)
    do_inferences(driver, infer_url, path_pred, column_names,
                  test_dict_ord)
    stop_server_and_driver(server, driver)
                  
                  
def get_accuracy(path_targets, path_pred, key_to_infer, path_accuracy):
    """
    Get the accuracy of your tests and save it to accuracy.txt.
    =Parameters=
    path_targets: The path where targets.txt is saved.
    path_pred: The path where predictions.txt is saved.
    key_to_infer: The column that you want to infer.
    path_accuracy: The path where you want to save accuracy.txt.
    """
    list_targets = []
    list_preds = []
    with open(path_targets, 'r') as f:
        for line in f.readlines():
            list_targets.append(line[:-1])
    with open(path_pred, 'r') as f:
        for line in f.readlines():
            list_preds.append(line[:-1])
            
    len_list = len(list_targets)
            
    if key_to_infer[1] == str or key_to_infer[1] == float:
        correct = 0
        incorrect = 0
        for idx in range(len_list):
            if list_targets[idx] == list_preds[idx]:
                correct +=1
            else:
                incorrect += 1
        with open(path_accuracy, 'w') as f:
            f.write("Test Accuracy:"+'\n')
            f.wirte("Total tests: {}".format(len_list)+'\n')
            f.write("Correct predictions: {}".format(str(correct))+'\n')
            f.write("Incorrect predictions: {}".format(str(incorrect))+'\n')
            f.write("Accuracy: {}".format(str(correct/len_list*100)+'%'))
            
    else:
        correct = 0
        incorrect = 0
        for idx in range(len_list):
            if int(list_targets[idx]) == int(round(float(list_preds[idx]))):
                correct +=1
            else:
                incorrect += 1
        with open(path_accuracy, 'w') as f:
            f.write("Test Accuracy:"+'\n')
            f.write("Total tests: {}".format(len_list)+'\n')
            f.write("Correct predictions: {}".format(str(correct))+'\n')
            f.write("Incorrect predictions: {}".format(str(incorrect))+'\n')
            f.write("Accuracy: {}".format(str(correct/float(len_list)*100)+'%'))
    

if __name__ == "__main__":
    run_test(PATH_TO_DRIVER, PATH_TO_BROWSER, PATH_TO_TEST, PATH_TARGETS, KEY_TO_INFER,
             COLUMNS_INT, COLUMNS_FLOAT, INFER_URL, PATH_PRED)
    get_accuracy(PATH_TARGETS, PATH_PRED, KEY_TO_INFER, PATH_ACC)






