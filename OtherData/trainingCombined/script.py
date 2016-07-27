all_examples = 'all_examples'
f = open(all_examples + '.txt', 'w')

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print len(letters)
for letter in letters:
  filename = letter
  curr = open(filename + '.txt')
  for line in curr:
    f.write(line)
  curr.close()
f.close()

