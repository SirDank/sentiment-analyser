import os
import nltk
import yaml
from pprint import pprint
from dankware import cls, clr, clr_banner, align, white, magenta, chdir
from flask import Flask, request, render_template

#exec(chdir("script"))

class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):

        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """

        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

class POSTagger(object):

    def __init__(self):
        pass
        
    def pos_tag(self, sentences):

        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary_paths):

        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.full_load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(str(key)))

    def tag(self, postagged_sentences):

        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):

        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """

        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N

        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

def value_of(sentiment):

    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):    

    if not sentence_tokens: return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):

    return sum([sentence_score(sentence, None, 0.0) for sentence in review])

def main():
    
    # clear screen and change working dir to filepath

    banner ="\n                                                                       \n             _   _               _                   _                 \n ___ ___ ___| |_|_|_____ ___ ___| |_ ___ ___ ___ ___| |_ _ ___ ___ ___ \n|_ -| -_|   |  _| |     | -_|   |  _|___| .'|   | .'| | | |_ -| -_|  _|\n|___|___|_|_|_| |_|_|_|_|___|_|_|_|     |__,|_|_|__,|_|_  |___|___|_|  \n                                                      |___|            \n\n"
    cls(); print(align(clr_banner(banner)))
    #os.chdir(os.path.dirname(__file__))
    
    # make files and folders

    try: os.mkdir('dicts'); print(clr('\n  > Created [ dicts ]'))
    except: pass
    
    try: open('dicts/dec.yml','x'); print(clr('\n  > Created [ dec.yml ]'))
    except: pass
    
    try: open('dicts/inc.yml','x'); print(clr('\n  > Created [ inc.yml ]'))
    except: pass
    
    try: open('dicts/inv.yml','x'); print(clr('\n  > Created [ inv.yml ]'))
    except: pass
    
    try: open('dicts/positive.yml','x'); print(clr('\n  > Created [ positive.yml ]'))
    except: pass
    
    try: open('dicts/negative.yml','x'); print(clr('\n  > Created [ negative.yml ]'))
    except: pass
    
    try: open('original_text.txt','x'); print(clr('\n  > Created [ original_text.txt ]'))
    except: pass
    
    try: open('summarized_text.txt','x'); print(clr('\n  > Created [ summarized_text.txt ]'))
    except: pass
    
    try: open('results.txt','x'); print(clr('\n  > Created [ results.txt ]'))
    except: pass
    
    #wait = input(clr("\n  > Add datasets to the 'dicts' folder\n\n  > Add the input text to 'original_text.txt'\n\n  > Add the summarized text to 'summarized_text.txt'\n\n  > Press [ ENTER ] after completing all of the above... "))
    
    # original text
    
    #run(open('original_text.txt','r').read().replace('\n',' '))
    
    # summarized text
    
    #run(open('summarized_text.txt','r').read().replace('\n',' '))
    
def run(text):

    cls(); print(clr("\n  > Pre-Processing Text..."))

    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml', 'dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'])
    
    print(clr("\n  > Splitting text into list of sentences and splitting sentences into list of words...\n"))
    splitted_sentences = splitter.split(text)
    pprint(splitted_sentences)
    
    print(clr("\n  > POS Tagging...\n"))
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    pprint(pos_tagged_sentences)

    print(clr("\n  > DICT Tagging...\n"))
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    pprint(dict_tagged_sentences)

    print(clr("\n  > Analyzing Sentiment..."))
    score = sentiment_score(dict_tagged_sentences)
    if score > 0: score = f"> Sentiment: Positive | Score: {score}"
    elif score < 0: score = f"> Sentiment: Negative | Score: {score}"
    else: score = f"> Sentiment: Neutral | Score: {score}"
    print(clr(f"\n  {score}\n"))
    
    # saving result to text file
    
    text = text.replace('\n',' ')
    result = f"{score} | {text}"
    open('results.txt','a').write(f"{result}\n")

    return(f"<br>  > Splitting text into list of sentences and splitting sentences into list of words...<br><br>{splitted_sentences}<br><br>  > POS Tagging...<br><br>{pos_tagged_sentences}<br><br>  > DICT Tagging...<br><br>{dict_tagged_sentences}<br><br>  {score}")

app = Flask(__name__)

@app.route('/')
def home():
    return "<pre><br>                                                                       <br>             _   _               _                   _                 <br> ___ ___ ___| |_|_|_____ ___ ___| |_ ___ ___ ___ ___| |_ _ ___ ___ ___ <br>|_ -| -_|   |  _| |     | -_|   |  _|___| .'|   | .'| | | |_ -| -_|  _|<br>|___|___|_|_|_| |_|_|_|_|___|_|_|_|     |__,|_|_|__,|_|_  |___|___|_|  <br>                                                      |___|            <br><br></pre>", 200

@app.route('/sentiment-analyser')
def sentiment_analyser_html():
    return render_template('input.html')

@app.route('/sentiment-analyser', methods=['POST'])
def sentiment_analyser():

    text = request.form['text']
    if text == None: return "<pre><br>  > Please Provide Input</pre>", 500
    else: return f"<pre>{run(text)}</pre>", 200

if __name__ == "__main__":

    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    except: pass

    main()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 3212)), threaded=True)
