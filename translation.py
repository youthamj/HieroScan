### Installs ###
# !pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# !pip install OpenNMT-py

### Imports ###
import subprocess


### START FUNCTION ###
def translate(model, transliteration_list):
    ### Save previous output into a text file ###
    input = open("input.txt", "w")
    for sentence in transliteration_list:
        input.write(sentence + "\n")
    input.flush()
    input.close()

    ### Translate and Save text ###
    input = 'input.txt'  # Input file path
    output = 'translation.txt'  # Output file path
    subprocess.run(
        'onmt_translate -model ' + model + ' -src ' + input + ' -output ' + output + ' -replace_unk -verbose',
        shell=True)

    ### Load text into list of strings ###
    # translation = [line.rstrip() for line in open(output, "r")]

    # return translation


### END FUNCTION ###


### Example ###
model = 'assets/translation_assets/_step_8000.pt'  # Model path
transliteration_list = ['pA rf nD q .f j  nTry idt t p Hnk k',
                        'd wnn .f aA aA .s mkt t  nn wn Hr xw .f n n wnt',
                        'rn js jrk nwn n .s j s p',
                        'n js p j imn rT msTw T sAT wnwn Twn',
                        'st swtwt y ']

translate(model, transliteration_list)
