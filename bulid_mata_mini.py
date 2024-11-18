import pandas as pd

# MetaData
metadata = pd.DataFrame(columns=['speakID', 'fileName', 'non', 'model', 'ANS'])

# read output PATH
file = open("path/output_path.txt", "r")
path = file.read()
file.close()


count = { 'm1':434, 'm2':444, 'm3':454, 'm4':449, 'm5':452 }
for (notation, count) in count.items() :
    for now_index in range(0, count+1) :

        filename = f"E03_{notation}_pmt_{now_index:06}.wav"
        metadata = metadata.loc[len(metadata)] = [notation, filename, '-', 'E03', 'spoof']

# E03 == parlerTTS
metadata.to_csv(f'{path}mini/E03_mini_spoof.csv', sep= ' ', index=False, header=False)