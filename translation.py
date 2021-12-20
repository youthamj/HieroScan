### Installs ###
# !pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# !pip install OpenNMT-py

### Imports ###
import subprocess

translation_model = 'assets/translation_assets/_step_8000.pt'  # Model path
### START FUNCTION ###
def translate(transliteration_list):
    global translation_model
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
        'onmt_translate -model ' + translation_model + ' -src ' + input + ' -output ' + output + ' -replace_unk -verbose',
        shell=True)

    ### Load text into list of strings ###
    translation = [line.rstrip() for line in open(output, "r")]

    return translation


### END FUNCTION ###

if __name__ == "__main__":
    ### Example ###

    # transliteration_list = ['pA rf nD q .f j  nTry idt t p Hnk k',
    #                         'd wnn .f aA aA .s mkt t  nn wn Hr xw .f n n wnt',
    #                         'rn js jrk nwn n .s j s p',
    #                         'n js p j imn rT msTw T sAT wnwn Twn',
    #                         'st swtwt y ']
    # transliteration_list = ['rf n nxt xrw q .f kApwt kApwt Ts t q Hnk', 'wnn .f aA aA .s A k rn  n t', 'Tn wab pA st qsnt rk nwn n  b bSTw', 'n p kA mnx ib rmnn rT msTw T']
    # transliteration_list = ['ngsgs  .f n Hnskt  q dwA wr dwA wr dwA wr n k n', 'n Hnskt dwA wr n g wnwty jj pn      ', 'bw k p k mnkrt wsi w wsi p gAp', 'Hnskt k k .f k j imn j srw .s smn Hnk k kft ', 'k m HAm d A wnwty wnwty dwA wr   ']
    # transliteration_list = ['smA', 'n k j .f mA - HsA .s H ir sp Hna m n  k', 't Apd Ax n k Hnskt r sqr  jty', 'xnr rk  nty .k  jmy r n', ' Hnskt k wnwty wnwty wnwty k   ']
    transliteration_list = ['pAxt p r hAyt r t .stAt t r xprS w H Hta r ', '.s sm sni n n n .s h rTn wab h',
                            'hyhy mwt rf b p pwnt n j h', 'm Smm n Tn Ax t nfr',
                            'Htpt p A jrj Dsrw jt jw w nTr dwAw nTry',
                            'Hnskt Hnn n k r jmw r w iw iwSS psS m xrw n p  pA  jw     Hww Swt   nwn n mn imn n',
                            'r wAt m ngmgm g d xnr n j Tms snaa ib']
    print(translate(transliteration_list))
