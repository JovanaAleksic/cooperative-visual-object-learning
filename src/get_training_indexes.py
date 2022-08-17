
def get_training_indexes(training_index, k):

    if training_index < 200:
        list=range(training_index, training_index+1000)

        test_list = range(training_index+1000, 1200)
        test_list.extend(range(training_index))
    else:
        list = range(training_index, 1200)
        list.extend(range(training_index-200))

        test_list = range(training_index-200, training_index)
    return list, test_list