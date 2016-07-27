training_text = 'training.txt'
training = open(training_text, 'w')

testing_text = 'testing.txt'
testing = open(testing_text, 'w')

letter_recognition_text = 'letter-recognition.txt'
letter_recognition = open(letter_recognition_text)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
letter_dict = dict()
for letter in letters:
  letter_dict[letter] = 0 # if divBy2, write current example to training file, 
                          # elif !divBy2 write to test file

for example in letter_recognition:
  #get target from current example
  target = example[0]

  if (letter_dict[target] % 2) is 0:
    training.write(example)
  elif (letter_dict[target] % 2) is 1:
    testing.write(example)

  #increment the count of the current example's target class
  letter_dict[target] += 1


letter_recognition.close()
training.close()
testing.close()


print letter_dict
