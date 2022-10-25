import os

train_dir = [f'data/obj/{file}' for file in os.listdir('obj') if file[-3:] == 'jpg']
valid_dir = [f'data/obj/{file}' for file in os.listdir('valid') if file[-3:] == 'jpg']

def writeToFile(fname, paths):
    with open(f'{fname}.txt', 'a') as file:
        for path in paths:
            file.write(f'{path}\n')
        else:
            file.close()

if __name__ == '__main__':
    writeToFile('train', train_dir)
    writeToFile('valid', valid_dir)

#%%

#%%
