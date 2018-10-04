# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:28:22 2016

@author: thinkwee
"""

import csv as csv
import numpy as np

test_file = (open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\test.csv', 'r'))
test_file_object = csv.reader(open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\test.csv', 'r'))
testheader = next(test_file_object)
predictions_file = open(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\genderbasedmodel.csv", "w")
predictions_file_object = csv.writer(predictions_file)
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])
csv_file_object = csv.reader(open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\train.csv', 'r'))
trainheader = next(csv_file_object)  # The next() command just skips the 
# first line which is a header
data = []  # Create a variable called 'data'.
for row in csv_file_object:  # Run through each row in the csv file,
    data.append(row)  # adding each row to the data variable
print(type(data))
data = np.array(data)  # Then convert from a list to an array
# Be aware that each item is currently
# a string in this format

number_passengers = np.size(data[0::, 1].astype(np.float))
number_survived = np.sum(data[0::, 1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[0::, 4] == "female"  # This finds where all
# the elements in the gender
# column that equals “female”
men_only_stats = data[0::, 4] != "female"  # This finds where all the
# elements do not equal
# female (i.e. male)

# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats, 1].astype(np.float)
men_onboard = data[men_only_stats, 1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = \
    np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = \
    np.sum(men_onboard) / np.size(men_onboard)

# and then print it out
print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)

# The script will systematically will loop through each combination and use the 'where' function in python to search the passengers that fit that combination of variables.
# Just like before, you can ask what indices in your data equals female, 1st class, and paid more than $30.
# The problem is that looping through requires bins of equal sizes, i.e. $0-9,  $10-19,  $20-29,  $30-39.
# For the sake of binning let's say everything equal to and above 40 "equals" 39 so it falls in this bin.
# So then you can set the bins

# So we add a ceiling
fare_ceiling = 40

# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[data[0::, 9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size

# Take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[0::, 2]))

number_of_age_brackets = 8

# Initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets, number_of_age_brackets))

# Now that these are set up,
# you can loop through each variable and find all those passengers that agree with the statements

for i in range(number_of_classes):  # loop through each class
    for j in range(number_of_price_brackets):  # loop through each price bin
        for k in range(number_of_age_brackets):  # loop through each age bin
            women_only_stats_plus = data[  # Which element
                (data[0::, 4] == "female")  # is a female
                & (data[0::, 2].astype(np.float)  # and was ith class
                   == i + 1)
                & (data[0:, 9].astype(np.float)  # was greater
                   >= j * fare_bracket_size)  # than this bin
                & (data[0:, 9].astype(np.float)  # and less than
                   < (j + 1) * fare_bracket_size)
                & (data[0:, 5].astype(np.float) >= k * 10)
                & (data[0:, 5].astype(np.float) < (k + 1) * 10)  # the next bin

                , 1]  # in the 2nd col

            men_only_stats_plus = data[  # Which element
                (data[0::, 4] != "female")  # is a male
                & (data[0::, 2].astype(np.float)  # and was ith class
                   == i + 1)
                & (data[0:, 9].astype(np.float)  # was greater
                   >= j * fare_bracket_size)  # than this bin
                & (data[0:, 9].astype(np.float)  # and less than
                   < (j + 1) * fare_bracket_size)  # the next bin
                & (data[0:, 5].astype(np.float) >= k * 10)
                & (data[0:, 5].astype(np.float) < (k + 1) * 10)
                , 1]

            survival_table[0, i, j, k] = np.mean(women_only_stats_plus.astype(np.float))
            survival_table[1, i, j, k] = np.mean(men_only_stats_plus.astype(np.float))
            survival_table[
                survival_table != survival_table] = 0.  # if nan then the type will change to string from float so this sentence can set nan to 0.

# Notice that  data[ where function, 1]  means it is finding the Survived column for the conditional criteria which is being called.
# As the loop starts with i=0 and j=0,
# the first loop will return the Survived values for all the 1st-class females (i + 1) who paid less than 10 ((j+1)*fare_bracket_size)
# and similarly all the 1st-class males who paid less than 10.
# Before resetting to the top of the loop,
# we can calculate the proportion of survivors for this particular combination of criteria and record it to our survival table



#    survival_table[ survival_table < 0.5 ] = 0
#    survival_table[ survival_table >= 0.5 ] = 1 


# Then we can make the prediction

for row in test_file_object:  # We are going to loop
    # through each passenger
    # in the test set
    for j in range(number_of_price_brackets):  # For each passenger we
        # loop thro each price bin
        try:  # Some passengers have no
            # Fare data so try to make
            row[8] = float(row[8])  # a float
        except:  # If fails: no data, so
            bin_fare = 3 - float(row[1])  # bin the fare according Pclass
            break  # Break from the loop
        if row[8] > fare_ceiling:  # If there is data see if
            # it is greater than fare
            # ceiling we set earlier
            bin_fare = number_of_price_brackets - 1  # If so set to highest bin
            break  # And then break loop
        if row[8] >= j * fare_bracket_size \
                and row[8] < \
                                (j + 1) * fare_bracket_size:  # If passed these tests
            # then loop through each bin
            bin_fare = j  # then assign index
            break

    for j in range(number_of_age_brackets):

        try:

            row[4] = float(row[4])
        except:
            bin_age = -1
            break

        if row[4] >= j * 10 \
                and row[4] < \
                                (j + 1) * 10:  # If passed these tests
            # then loop through each bin
            bin_age = j  # then assign index
            break

    if row[3] == 'female':  # If the passenger is female
        p.writerow([row[0], "%f %%" % \
                    (survival_table[0, int(row[1]) - 1, bin_fare, bin_age] * 100)])
    else:  # else if male
        p.writerow([row[0], "%f %%" % \
                    (survival_table[1, int(row[1]) - 1, bin_fare, bin_age] * 100)])

# Close out the files.
test_file.close()
predictions_file.close()
