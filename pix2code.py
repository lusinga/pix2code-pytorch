import pdb
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image

# Hyperparams
batch_size = 4
embed_size = 256
num_epochs = 1000
learning_rate = 0.001
hidden_size = 512
num_layers = 1

# Other params
shuffle = True
num_workers = 2

# Logging Variables
save_after_x_epochs = 50
log_step = 5

# Paths
data_dir = './processed_data/data_train/' # For testing purposes, we use a pre-split dataset rather than do it here
model_path = './models/'
vocab_path = './bootstrap.vocab'

# DO NOT CHANGE:
crop_size = 224 # Required by resnet152

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

class Vocabulary (object):
    def __init__ (self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word (self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__ (self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__ (self):
        return len(self.word2idx)

def build_vocab (vocab_file_path):
    vocab = Vocabulary()

    # Load the vocab file (super basic split())
    words_raw = load_doc(vocab_file_path)
    words = set(words_raw.split(' '))
    
    for i, word in enumerate(words):
        vocab.add_word(word)

    vocab.add_word(' ')
    vocab.add_word('<unk>') # If we find an unknown word
    
    print('Created vocabulary of ' + str(len(vocab)) + ' items from ' + vocab_file_path)
    
    return vocab

# Load vocabulary
vocab = build_vocab(vocab_path)

vocab_size = len(vocab)

class ImageHTMLDataSet (Dataset):
    def __init__ (self, data_dir, vocab, transform):
        self.data_dir = data_dir
        self.vocab = vocab
        self.transform = transform
        
        self.raw_image_names = []
        self.raw_captions = []
        
        # Fetch all files from our data directoruy
        self.filenames = os.listdir(data_dir)
        self.filenames.sort()
        
        # Sort files based on their filetype
        # Assume associated training examples have same filenames
        for filename in self.filenames:
            if filename[-3:] == 'png':
                # Store image filename
                self.raw_image_names.append(filename)
            elif filename[-3:] == 'gui':
                # Load .gui file
                data = load_doc(data_dir + filename)
                self.raw_captions.append(data)
                
        print('Created dataset of ' + str(len(self)) + ' items from ' + data_dir)

    def __len__ (self):
        return len(self.raw_image_names)
    
    def __getitem__ (self, idx):
        img_path, raw_caption = self.raw_image_names[idx], self.raw_captions[idx]
        
        # Get image from filesystem
        image = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
        image = self.transform(image)
        
        # Convert caption (string) to list of vocab ID's
        caption = []
        caption.append(self.vocab('<START>'))
        
        # Remove newlines, separate words with spaces
        tokens = ' '.join(raw_caption.split())

        # Add space after each comma
        tokens = tokens.replace(',', ' ,')
        
        # Split into words
        tokens = tokens.split(' ')
        
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<END>'))
        
        target = torch.Tensor(caption)
        
        return image, target

# See https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
def collate_fn (data):
    # Sort datalist by caption length; descending order
    data.sort(key = lambda data_pair: len(data_pair[1]), reverse=True)
    images, captions = zip(*data)
    
    # Merge images (from tuple of 3D Tensor to 4D Tensor)
    images = torch.stack(images, 0)
    
    # Merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(caption) for caption in captions] # List of caption lengths
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for i, caption in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = caption[:end]
        
    return images, targets, lengths

# Transform to modify images for pre-trained ResNet base
transform = transforms.Compose([
    transforms.Resize((crop_size, crop_size)), # Match resnet size
    transforms.ToTensor(),
    # See for magic #'s: http://pytorch.org/docs/master/torchvision/models.html
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create data loader
img_html_dataset = ImageHTMLDataSet(data_dir=data_dir, vocab=vocab, transform=transform)
data_loader = DataLoader(dataset=img_html_dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         collate_fn=collate_fn)


class EncoderCNN (nn.Module):
    def __init__ (self, embed_size):
        super(EncoderCNN, self).__init__()
        
        # Load pretrained resnet model
        resnet = models.resnet152(pretrained = True)
        
        # Remove the fully connected layers
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Create our replacement layers
        # We reuse the in_feature size of the resnet fc layer for our first replacement layer = 2048 as of creation
        self.linear = nn.Linear(in_features = resnet.fc.in_features, out_features = embed_size)
        self.bn = nn.BatchNorm1d(num_features = embed_size, momentum = 0.01)
        
        print('EncoderCNN created with embed_size: ' + str(embed_size))

    def forward (self, images):
        # Get the expected output from the fully connected layers
        # Fn: AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        # Output: torch.Size([batch_size, 2048, 1, 1])
        features = self.resnet(images)

        # Resize the features for our linear function
        features = features.view(features.size(0), -1)
        
        # Fn: Linear(in_features=2048, out_features=embed_size, bias=True)
        # Output: torch.Size([batch_size, embed_size])
        features = self.linear(features)
        
        # Fn: BatchNorm1d(embed_size, eps=1e-05, momentum=0.01, affine=True)
        # Output: torch.Size([batch_size, embed_size])
        features = self.bn(features)
        
        return features

class DecoderRNN (nn.Module):
    def __init__ (self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        
        # 19 word vocabulary, embed_size dimensional embeddings
        self.embed = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)

        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

        self.linear = nn.Linear(in_features = hidden_size, out_features = vocab_size)
        
        # Store the embed size for use when sampling
        self.embed_size = embed_size
        
        print('DecoderRNN created with embed_size: ' + str(embed_size))
        
    def forward (self, features, captions, lengths):
        # 'captions' enters as shape torch.Size([batch_size, len(longest caption)])
        
        # Fn: Embedding(vocab_size, embed_size)
        # Input: LongTensor (N = mini_batch, W = # of indices to extract per mini-batch)
        # Output: (N, W, embedding_dim) => torch.Size([batch_size, len(longest caption), embed_size])
        embeddings = self.embed(captions)
        
        # Match features dimensions to embedding's
        features = features.unsqueeze(1) # torch.Size([4, 128]) => torch.Size([4, 1, 128])
        
        embeddings = torch.cat((features, embeddings), 1)
        
        packed = nn.utils.rnn.pack_padded_sequence(input = embeddings, lengths = lengths, batch_first = True)
        
        # Fn: LSTM(embed_size, hidden_size, batch_first = True)
        hiddens, _ = self.lstm(packed)
        
        outputs = self.linear(hiddens[0])
        
        return outputs

    # Sample method used for testing our model
    def sample (self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        # Put the features input through our decoder for i iterations
        # TODO: Put this range into a parameter?
        for i in range(100):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(dim = 1, keepdim = True)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.view(-1, 1, self.embed_size)

        sampled_ids = torch.cat(sampled_ids, 1)

        return sampled_ids.squeeze()

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr = learning_rate)

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    print('CUDA activated.')

encoder.train()
decoder.train()

batch_count = len(data_loader)

for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(data_loader):
        # Shape: torch.Size([batch_size, 3, crop_size, crop_size])
        images = Variable(images.cuda())

        # Shape: torch.Size([batch_size, len(longest caption)])
        captions = Variable(captions.cuda())

        # lengths is a list of how long captions are in descending order (e.g., [77, 77, 75, 25])

        # We remove the paddings from captions that are padded and then pack them into a single sequence
        # Our data loader's collate_fn adds extra zeros to the end of sequences that are too short
        # Shape: torch.Size([sum(lengths)])
        targets = nn.utils.rnn.pack_padded_sequence(input = captions, lengths = lengths, batch_first = True)[0]

        # Zero out buffers
        encoder.zero_grad()
        decoder.zero_grad()

        # Forward, Backward, and Optimize
        features = encoder(images) # Outputs features of torch.Size([batch_size, embed_size])
        outputs = decoder(features, captions, lengths)

        # CrossEntropyLoss is expecting:
        # Input:  (N, C) where C = number of classes
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()


        if epoch % log_step == 0 and i == 0:
            print('Epoch [#%d], Loss: %.4f, Perplexity: %5.4f'
                  % (epoch, loss.data[0], np.exp(loss.data[0])))
            
        if (epoch + 1) % save_after_x_epochs == 0 and i == 0:
            # Save our models
            print('!!! saving models at epoch: ' + str(epoch))
            torch.save(decoder.state_dict(),os.path.join(model_path, 'decoder-%d-%d.pkl' %(epoch+1, i+1)))
            torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-%d-%d.pkl' %(epoch+1, i+1)))
            
print('done!')

torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-1000-1.pkl'))
torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder-1000-1.pkl'))
