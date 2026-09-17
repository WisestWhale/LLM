class Tokenizer:
    def __init__(self , text):

      unique = self.Preprocessed(text)
      
      unique_tokens = sorted(list(set(unique)))
      unique_tokens.append("<|unk|>")
      self.word_to_id = {word: i for i , word in enumerate (unique_tokens)}
      self.id_to_word = {i: word for i , word in enumerate (unique_tokens)}


    def Preprocessed(self , text):
      import re
      tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
      return [item.strip() for item in tokens if item.strip()]
      
      


    def encode(self , text):
        
        new_tokens = self.Preprocessed(text)
        unk_id = self.word_to_id["<|unk|>"]
        return [self.word_to_id.get(word, unk_id) for word in new_tokens ]
    
    def decode(self , tokens_id):
        
        words = [self.id_to_word[tokens] for tokens in tokens_id]
        return " ".join(words)



