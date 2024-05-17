{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: \u001b[33mWARNING\u001b[0m If you're specifying your api key in code, ensure this code is not shared publicly.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \u001b[33mWARNING\u001b[0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: Appending key for api.wandb.ai to your netrc file: C:\\Users\\Amlan\\.netrc\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "import random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import time\n",
    "\n",
    "import wandb\n",
    "wandb.login(key = '8cd670a52fc28bf254ff6ff2b01f010982869e8d')\n",
    "\n",
    "random.seed()\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Language Model\n",
    "SOS_token = 0\n",
    "EOS_token = 1\n",
    "\n",
    "class Language:\n",
    "    def __init__(self, name):\n",
    "        self.name = name\n",
    "        self.word2index = {}\n",
    "        self.word2count = {}\n",
    "        self.index2word = {SOS_token: \"<\", EOS_token: \">\"}\n",
    "        self.n_chars = 2  # Count SOS and EOS\n",
    "\n",
    "    def addWord(self, word):\n",
    "        for char in word:\n",
    "            self.addChar(char)\n",
    "\n",
    "    def addChar(self, char):\n",
    "        if char not in self.word2index:\n",
    "            self.word2index[char] = self.n_chars\n",
    "            self.word2count[char] = 1\n",
    "            self.index2word[self.n_chars] = char\n",
    "            self.n_chars += 1\n",
    "        else:\n",
    "            self.word2count[char] += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_data(lang: str, type: str) -> list[list[str]]:\n",
    "    \"\"\"\n",
    "    Returns: 'pairs': list of [input_word, target_word] pairs\n",
    "    \"\"\"\n",
    "    path = \"aksharantar_sampled/aksharantar_sampled/{}/{}_{}.csv\".format(lang, lang, type)\n",
    "    df = pd.read_csv(path, header=None)\n",
    "    pairs = df.values.tolist()\n",
    "    return pairs"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "get_data('hin', 'train')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_languages(lang: str):\n",
    "    \"\"\"\n",
    "    Returns \n",
    "    1. input_lang: input language - English\n",
    "    2. output_lang: output language - Given language\n",
    "    3. pairs: list of [input_word, target_word] pairs\n",
    "    \"\"\"\n",
    "    input_lang = Language('eng')\n",
    "    output_lang = Language(lang)\n",
    "    pairs = get_data(lang, \"train\")\n",
    "    for pair in pairs:\n",
    "        input_lang.addWord(pair[0])\n",
    "        output_lang.addWord(pair[1])\n",
    "    return input_lang, output_lang, pairs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_cell(cell_type: str):\n",
    "    if cell_type == \"LSTM\":\n",
    "        return nn.LSTM\n",
    "    elif cell_type == \"GRU\":\n",
    "        return nn.GRU\n",
    "    elif cell_type == \"RNN\":\n",
    "        return nn.RNN\n",
    "    else:\n",
    "        raise Exception(\"Invalid cell type\")\n",
    "    \n",
    "def get_optimizer(optimizer: str):\n",
    "    if optimizer == \"SGD\":\n",
    "        return optim.SGD\n",
    "    elif optimizer == \"ADAM\":\n",
    "        return optim.Adam\n",
    "    else:\n",
    "        raise Exception(\"Invalid optimizer\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Encoder(nn.Module):\n",
    "    def __init__(self,\n",
    "                 in_sz: int,\n",
    "                 embed_sz: int,\n",
    "                 hidden_sz: int,\n",
    "                 cell_type: str,\n",
    "                 n_layers: int,\n",
    "                 dropout: float):\n",
    "        \n",
    "        super(Encoder, self).__init__()\n",
    "        self.hidden_sz = hidden_sz\n",
    "        self.n_layers = n_layers\n",
    "        self.dropout = dropout\n",
    "        self.cell_type = cell_type\n",
    "        self.embedding = nn.Embedding(in_sz, embed_sz)\n",
    "\n",
    "        self.rnn = get_cell(cell_type)(input_size = embed_sz,\n",
    "                                       hidden_size = hidden_sz,\n",
    "                                       num_layers = n_layers,\n",
    "                                       dropout = dropout)\n",
    "        \n",
    "    def forward(self, input, hidden, cell):\n",
    "        embedded = self.embedding(input).view(1, 1, -1)\n",
    "\n",
    "        if(self.cell_type == \"LSTM\"):\n",
    "            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))\n",
    "        else:\n",
    "            output, hidden = self.rnn(embedded, hidden)\n",
    "            \n",
    "        return output, hidden, cell\n",
    "    \n",
    "    def initHidden(self):\n",
    "        return torch.zeros(self.n_layers, 1, self.hidden_sz, device=device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Decoder(nn.Module):\n",
    "    def __init__(self,\n",
    "                 out_sz: int,\n",
    "                 embed_sz: int,\n",
    "                 hidden_sz: int,\n",
    "                 cell_type: str,\n",
    "                 n_layers: int,\n",
    "                 dropout: float):\n",
    "\n",
    "        super(Decoder, self).__init__()\n",
    "        self.hidden_sz = hidden_sz\n",
    "        self.n_layers = n_layers\n",
    "        self.dropout = dropout\n",
    "        self.cell_type = cell_type\n",
    "        self.embedding = nn.Embedding(out_sz, embed_sz)\n",
    "\n",
    "        self.rnn = get_cell(cell_type)(input_size = embed_sz,\n",
    "                                        hidden_size = hidden_sz,\n",
    "                                        num_layers = n_layers,\n",
    "                                        dropout = dropout)\n",
    "        \n",
    "        self.out = nn.Linear(hidden_sz, out_sz)\n",
    "        self.softmax = nn.LogSoftmax(dim=1)\n",
    "\n",
    "    def forward(self, input, hidden, cell):\n",
    "        output = self.embedding(input).view(1, 1, -1)\n",
    "        output = F.relu(output)\n",
    "\n",
    "        if(self.cell_type == \"LSTM\"):\n",
    "            output, (hidden, cell) = self.rnn(output, (hidden, cell))\n",
    "        else:\n",
    "            output, hidden = self.rnn(output, hidden)\n",
    "            \n",
    "        output = self.softmax(self.out(output[0]))\n",
    "        return output, hidden, cell\n",
    "    \n",
    "    def initHidden(self):\n",
    "        return torch.zeros(self.n_layers, 1, self.hidden_sz, device=device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "def indexesFromWord(lang:Language, word:str):\n",
    "    return [lang.word2index[char] for char in word]\n",
    "\n",
    "def tensorFromWord(lang:Language, word:str):\n",
    "    indexes = indexesFromWord(lang, word)\n",
    "    indexes.append(EOS_token)\n",
    "    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)\n",
    "\n",
    "def tensorsFromPair(input_lang:Language, output_lang:Language, pair:list[str]):\n",
    "    input_tensor = tensorFromWord(input_lang, pair[0])\n",
    "    target_tensor = tensorFromWord(output_lang, pair[1])\n",
    "    return (input_tensor, target_tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "def params_definition(): \n",
    "    \"\"\"\n",
    "    params:\n",
    "\n",
    "        embed_size : size of embedding (input and output) (8, 16, 32, 64)\n",
    "        hidden_size : size of hidden layer (64, 128, 256, 512)\n",
    "        cell_type : type of cell (LSTM, GRU, RNN)\n",
    "        num_layers : number of layers in encoder (1, 2, 3)\n",
    "        dropout : dropout probability\n",
    "        learning_rate : learning rate\n",
    "        teacher_forcing_ratio : teacher forcing ratio (0.5 fixed for now)\n",
    "        optimizer : optimizer (SGD, Adam)\n",
    "        max_length : maximum length of input word (50 fixed for now)\n",
    "\n",
    "    \"\"\"\n",
    "    pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "PRINT_EVERY = 40000\n",
    "PLOT_EVERY = 40000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Translator:\n",
    "    def __init__(self, lang: str, params: dict):\n",
    "        self.lang = lang\n",
    "        self.input_lang, self.output_lang, self.pairs = get_languages(self.lang)\n",
    "        self.input_size = self.input_lang.n_chars\n",
    "        self.output_size = self.output_lang.n_chars\n",
    "\n",
    "        self.training_pairs = [tensorsFromPair(self.input_lang, self.output_lang, pair) for pair in self.pairs]\n",
    "\n",
    "        self.encoder = Encoder(in_sz = self.input_size,\n",
    "                             embed_sz = params[\"embed_size\"],\n",
    "                             hidden_sz = params[\"hidden_size\"],\n",
    "                             cell_type = params[\"cell_type\"],\n",
    "                             n_layers = params[\"num_layers\"],\n",
    "                             dropout = params[\"dropout\"]).to(device)\n",
    "        \n",
    "        self.decoder = Decoder(out_sz = self.output_size,\n",
    "                             embed_sz = params[\"embed_size\"],\n",
    "                             hidden_sz = params[\"hidden_size\"],\n",
    "                             cell_type = params[\"cell_type\"],\n",
    "                             n_layers = params[\"num_layers\"],\n",
    "                             dropout = params[\"dropout\"]).to(device)\n",
    "\n",
    "        self.encoder_optimizer = get_optimizer(params[\"optimizer\"])(self.encoder.parameters(), lr=params[\"learning_rate\"])\n",
    "        self.decoder_optimizer = get_optimizer(params[\"optimizer\"])(self.decoder.parameters(), lr=params[\"learning_rate\"])\n",
    "        \n",
    "        self.criterion = nn.NLLLoss()\n",
    "\n",
    "        self.teacher_forcing_ratio = params[\"teacher_forcing_ratio\"]\n",
    "        self.max_length = params[\"max_length\"]\n",
    "\n",
    "    def train_single(self, input_tensor, target_tensor):\n",
    "        encoder_hidden = self.encoder.initHidden()\n",
    "        encoder_cell = self.encoder.initHidden()\n",
    "\n",
    "        self.encoder_optimizer.zero_grad()\n",
    "        self.decoder_optimizer.zero_grad()\n",
    "\n",
    "        input_length = input_tensor.size(0)\n",
    "        target_length = target_tensor.size(0)\n",
    "\n",
    "        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_sz, device=device)\n",
    "\n",
    "        loss = 0\n",
    "\n",
    "        for ei in range(input_length):\n",
    "            encoder_output, encoder_hidden, encoder_cell = self.encoder(input_tensor[ei], encoder_hidden, encoder_cell)\n",
    "            encoder_outputs[ei] = encoder_output[0, 0]\n",
    "\n",
    "        decoder_input = torch.tensor([[SOS_token]], device=device)\n",
    "        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell\n",
    "\n",
    "        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False\n",
    "\n",
    "        if use_teacher_forcing:\n",
    "            for di in range(target_length):\n",
    "                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)\n",
    "                loss += self.criterion(decoder_output, target_tensor[di])\n",
    "\n",
    "                decoder_input = target_tensor[di]\n",
    "        else:\n",
    "            for di in range(target_length):\n",
    "                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)\n",
    "                loss += self.criterion(decoder_output, target_tensor[di])\n",
    "\n",
    "                topv, topi = decoder_output.topk(1)\n",
    "                decoder_input = topi.squeeze().detach()\n",
    "                if decoder_input.item() == EOS_token:\n",
    "                    break\n",
    "\n",
    "        loss.backward()\n",
    "        self.encoder_optimizer.step()\n",
    "        self.decoder_optimizer.step()\n",
    "\n",
    "        return loss.item() / target_length\n",
    "    \n",
    "    def train(self, iters=-1):\n",
    "        start_time = time.time()\n",
    "        plot_losses = []\n",
    "        print_loss_total = 0\n",
    "        plot_loss_total = 0\n",
    "\n",
    "        random.shuffle(self.training_pairs)\n",
    "        iters = len(self.training_pairs) if iters == -1 else iters\n",
    "\n",
    "        for iter in range(1, iters+1):\n",
    "            training_pair = self.training_pairs[iter - 1]\n",
    "            input_tensor = training_pair[0]\n",
    "            target_tensor = training_pair[1]\n",
    "\n",
    "            loss = self.train_single(input_tensor, target_tensor)\n",
    "            print_loss_total += loss\n",
    "            plot_loss_total += loss\n",
    "\n",
    "            if iter % PRINT_EVERY == 0:\n",
    "                print_loss_avg = print_loss_total / PRINT_EVERY\n",
    "                print_loss_total = 0\n",
    "                current_time = time.time()\n",
    "                print(\"Loss: {:.4f} | Iterations: {} | Time: {:.3f}\".format(print_loss_avg, iter, current_time - start_time))\n",
    "\n",
    "            if iter % PLOT_EVERY == 0:\n",
    "                plot_loss_avg = plot_loss_total / PLOT_EVERY\n",
    "                plot_losses.append(plot_loss_avg)\n",
    "                plot_loss_total = 0\n",
    "            \n",
    "        return plot_losses\n",
    "    \n",
    "    def evaluate(self, word):\n",
    "        with torch.no_grad():\n",
    "            input_tensor = tensorFromWord(self.input_lang, word)\n",
    "            input_length = input_tensor.size()[0]\n",
    "            encoder_hidden = self.encoder.initHidden()\n",
    "            encoder_cell = self.encoder.initHidden()\n",
    "\n",
    "            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_sz, device=device)\n",
    "\n",
    "            for ei in range(input_length):\n",
    "                encoder_output, encoder_hidden, encoder_cell = self.encoder(input_tensor[ei], encoder_hidden, encoder_cell)\n",
    "                encoder_outputs[ei] += encoder_output[0, 0]\n",
    "\n",
    "            decoder_input = torch.tensor([[SOS_token]], device=device)\n",
    "            decoder_hidden, decoder_cell = encoder_hidden, encoder_cell\n",
    "\n",
    "            decoded_chars = \"\"\n",
    "\n",
    "            for di in range(self.max_length):\n",
    "                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)\n",
    "                topv, topi = decoder_output.topk(1)\n",
    "                \n",
    "                if topi.item() == EOS_token:\n",
    "                    break\n",
    "                else:\n",
    "                    decoded_chars += self.output_lang.index2word[topi.item()]\n",
    "\n",
    "                decoder_input = topi.squeeze().detach()\n",
    "\n",
    "            return decoded_chars\n",
    "        \n",
    "    def test_validate(self, type:str):\n",
    "        pairs = get_data(self.lang, type)\n",
    "        accuracy = np.sum([self.evaluate(pair[0]) == pair[1] for pair in pairs])\n",
    "        return accuracy / len(pairs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_losses(plot_losses, title: str):\n",
    "    # return plot of losses\n",
    "    x_labels = [i * PLOT_EVERY for i in range(1, len(plot_losses) + 1)]\n",
    "    plt.plot(x_labels, plot_losses, color=\"blue\")\n",
    "    plt.xlabel(\"Iterations\")\n",
    "    plt.ylabel(\"Loss\")\n",
    "    plt.title(title)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "sweep_configuration = {\n",
    "    \"method\": \"bayes\",\n",
    "    \"name\" : \"valo_sweep\",\n",
    "    \"metric\": {\n",
    "        \"name\": \"validation_accuracy\",\n",
    "        \"goal\": \"maximize\"\n",
    "    },\n",
    "    \"parameters\": {\n",
    "        \"embed_size\": {\n",
    "            \"values\": [8, 16, 32]\n",
    "        },\n",
    "        \"hidden_size\": {\n",
    "            \"values\": [128, 256, 512]\n",
    "        },\n",
    "        \"cell_type\": {\n",
    "            \"values\": [\"GRU\", \"LSTM\", \"RNN\"]\n",
    "        },\n",
    "        \"num_layers\": {\n",
    "            \"values\": [1, 2, 3]\n",
    "        },\n",
    "        \"dropout\": {\n",
    "            \"values\": [0, 0.1, 0.2]\n",
    "        },\n",
    "        \"learning_rate\": {\n",
    "            \"values\": [0.0005, 0.001, 0.005]\n",
    "        },\n",
    "        \"optimizer\": {\n",
    "            \"values\": [\"SGD\", \"ADAM\"]\n",
    "        },\n",
    "        \"teacher_forcing_ratio\": {\n",
    "            'value': 0.5\n",
    "        },\n",
    "        \"max_length\": {\n",
    "            'value': 50\n",
    "        }\n",
    "    }\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "count = 0\n",
    "\n",
    "def train_sweep():\n",
    "    global count\n",
    "    count += 1\n",
    "\n",
    "    run = wandb.init()\n",
    "    config = wandb.config\n",
    "    run.name = \"embed_size: {} | hidden_size: {} | cell_type: {} | num_layers: {} | dropout: {} | learning_rate: {} | optimizer: {} | teacher_forcing_ratio: {} | max_length: {}\".format(config.embed_size, config.hidden_size, config.cell_type, config.num_layers, config.dropout, config.learning_rate, config.optimizer, config.teacher_forcing_ratio, config.max_length)\n",
    "\n",
    "    model = Translator('hin', config)\n",
    "\n",
    "    epochs = 5\n",
    "    old_validation_accuracy = 0\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        print(\"Epoch: {}\".format(epoch + 1))\n",
    "        plot_losses = model.train()\n",
    "\n",
    "        # take average of plot losses as training loss\n",
    "        training_loss = sum(plot_losses) / len(plot_losses)\n",
    "        \n",
    "#         training_accuracy = model.test_validate('train')\n",
    "#         print(\"Training Accuracy: {:.4f}\".format(training_accuracy))\n",
    "\n",
    "        validation_accuracy = model.test_validate('valid')\n",
    "        # print(\"Validation Accuracy: {:.4f}\".format(validation_accuracy))\n",
    "\n",
    "        wandb.log({\n",
    "            \"epoch\": epoch + 1,\n",
    "            \"training_loss\": training_loss,\n",
    "            # \"training_accuracy\": training_accuracy,\n",
    "            \"validation_accuracy\": validation_accuracy\n",
    "        })\n",
    "\n",
    "        if epoch > 0:\n",
    "            if validation_accuracy < 0.0001:\n",
    "                break\n",
    "\n",
    "            if validation_accuracy < 0.95 * old_validation_accuracy:\n",
    "                break\n",
    "\n",
    "        old_validation_accuracy = validation_accuracy\n",
    "\n",
    "    test_accuracy = model.test_validate('test')\n",
    "    print(\"Test Accuracy: {:.4f}\".format(test_accuracy))\n",
    "\n",
    "    wandb.log({\n",
    "        \"test_accuracy\": test_accuracy\n",
    "    })\n",
    "\n",
    "    run.finish()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Create sweep with ID: jaevdu4x\n",
      "Sweep URL: https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: bhf9w6ck with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: RNN\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.1\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 8\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 512\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.005\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: SGD\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240516_195355-bhf9w6ck</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/bhf9w6ck' target=\"_blank\">woven-sweep-1</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/bhf9w6ck' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/bhf9w6ck</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Loss: 2.6977 | Iterations: 40000 | Time: 914.910\n",
      "Epoch: 2\n",
      "Loss: 2.5628 | Iterations: 40000 | Time: 901.986\n",
      "Epoch: 3\n",
      "Loss: 2.6913 | Iterations: 40000 | Time: 918.330\n",
      "Epoch: 4\n",
      "Loss: 2.8614 | Iterations: 40000 | Time: 910.734\n",
      "Test Accuracy: 0.0000\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.001 MB of 0.001 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>▄▁▄█</td></tr><tr><td>validation_accuracy</td><td>▁▁█▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>4</td></tr><tr><td>test_accuracy</td><td>0.0</td></tr><tr><td>training_loss</td><td>2.8614</td></tr><tr><td>validation_accuracy</td><td>0.00024</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">woven-sweep-1</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/bhf9w6ck' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/bhf9w6ck</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>.\\wandb\\run-20240516_195355-bhf9w6ck\\logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Sweep Agent: Waiting for job.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: Job received.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: mxhqpntk with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: LSTM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.1\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 32\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 512\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.001\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 3\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240516_211433-mxhqpntk</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/mxhqpntk' target=\"_blank\">magic-sweep-2</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/mxhqpntk' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/mxhqpntk</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Loss: 2.2577 | Iterations: 40000 | Time: 1570.908\n",
      "Epoch: 2\n",
      "Loss: 1.3890 | Iterations: 40000 | Time: 1480.693\n",
      "Epoch: 3\n",
      "Loss: 1.0154 | Iterations: 40000 | Time: 1605.248\n",
      "Epoch: 4\n",
      "Loss: 0.9019 | Iterations: 40000 | Time: 1580.483\n",
      "Epoch: 5\n",
      "Loss: 0.8466 | Iterations: 40000 | Time: 7206.955\n",
      "Test Accuracy: 0.2397\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.001 MB of 0.001 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▄▂▁▁</td></tr><tr><td>validation_accuracy</td><td>▁▅▇██</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>5</td></tr><tr><td>test_accuracy</td><td>0.23975</td></tr><tr><td>training_loss</td><td>0.84661</td></tr><tr><td>validation_accuracy</td><td>0.25903</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">magic-sweep-2</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/mxhqpntk' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/mxhqpntk</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>.\\wandb\\run-20240516_211433-mxhqpntk\\logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: f7pil6av with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: RNN\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 32\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 512\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.001\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: SGD\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_014211-f7pil6av</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f7pil6av' target=\"_blank\">dazzling-sweep-3</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f7pil6av' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f7pil6av</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Loss: 2.5874 | Iterations: 40000 | Time: 1515.413\n",
      "Epoch: 2\n",
      "Loss: 1.8869 | Iterations: 40000 | Time: 1583.985\n",
      "Epoch: 3\n",
      "Loss: 1.5882 | Iterations: 40000 | Time: 1582.113\n",
      "Epoch: 4\n",
      "Loss: 1.4337 | Iterations: 40000 | Time: 1587.753\n",
      "Epoch: 5\n",
      "Loss: 1.3427 | Iterations: 40000 | Time: 1587.803\n",
      "Test Accuracy: 0.0952\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.001 MB of 0.001 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▄▂▂▁</td></tr><tr><td>validation_accuracy</td><td>▁▃▅██</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>5</td></tr><tr><td>test_accuracy</td><td>0.09521</td></tr><tr><td>training_loss</td><td>1.34272</td></tr><tr><td>validation_accuracy</td><td>0.10059</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">dazzling-sweep-3</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f7pil6av' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f7pil6av</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>.\\wandb\\run-20240517_014211-f7pil6av\\logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: 51afibst with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: RNN\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 8\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 256\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.001\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 1\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_043558-51afibst</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/51afibst' target=\"_blank\">dark-sweep-4</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/51afibst' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/51afibst</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Loss: 2.8420 | Iterations: 40000 | Time: 848.136\n",
      "Epoch: 2\n",
      "Loss: 2.8382 | Iterations: 40000 | Time: 848.736\n",
      "Test Accuracy: 0.0000\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.001 MB of 0.006 MB uploaded\\r'), FloatProgress(value=0.1986754966887417, max=1.0…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▁</td></tr><tr><td>validation_accuracy</td><td>▁▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>2</td></tr><tr><td>test_accuracy</td><td>0.0</td></tr><tr><td>training_loss</td><td>2.83821</td></tr><tr><td>validation_accuracy</td><td>0.0</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">dark-sweep-4</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/51afibst' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/51afibst</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>.\\wandb\\run-20240517_043558-51afibst\\logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: xle53406 with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: LSTM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 32\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 512\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.001\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_051412-xle53406</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/xle53406' target=\"_blank\">winter-sweep-5</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/xle53406' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/xle53406</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Loss: 1.5071 | Iterations: 40000 | Time: 2189.618\n",
      "Epoch: 2\n",
      "Loss: 0.8901 | Iterations: 40000 | Time: 2075.710\n",
      "Epoch: 3\n",
      "Loss: 0.7937 | Iterations: 40000 | Time: 2075.268\n",
      "Epoch: 4\n",
      "Loss: 0.7431 | Iterations: 40000 | Time: 2075.076\n",
      "Epoch: 5\n",
      "Loss: 0.7093 | Iterations: 40000 | Time: 2073.576\n",
      "Test Accuracy: 0.2876\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "55b7be825f724963be1b6ee5c789ba1f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.001 MB of 0.001 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▃▂▁▁</td></tr><tr><td>validation_accuracy</td><td>▁▅▆▇█</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>5</td></tr><tr><td>test_accuracy</td><td>0.2876</td></tr><tr><td>training_loss</td><td>0.70933</td></tr><tr><td>validation_accuracy</td><td>0.31104</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">winter-sweep-5</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/xle53406' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/xle53406</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>.\\wandb\\run-20240517_051412-xle53406\\logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: 3en46bfd with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: RNN\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 32\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 512\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.0005\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_090300-3en46bfd</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/3en46bfd' target=\"_blank\">firm-sweep-6</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/3en46bfd' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/3en46bfd</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Loss: 2.8596 | Iterations: 40000 | Time: 1430.772\n",
      "Epoch: 2\n",
      "Loss: 2.8507 | Iterations: 40000 | Time: 1435.263\n",
      "Test Accuracy: 0.0000\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a12cc8700ed3458bb52528562d640e91",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.001 MB of 0.001 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▁</td></tr><tr><td>validation_accuracy</td><td>▁▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>2</td></tr><tr><td>test_accuracy</td><td>0.0</td></tr><tr><td>training_loss</td><td>2.85072</td></tr><tr><td>validation_accuracy</td><td>0.0</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">firm-sweep-6</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/3en46bfd' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/3en46bfd</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>.\\wandb\\run-20240517_090300-3en46bfd\\logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: dedc0ee5 with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: LSTM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.1\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 32\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 512\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.001\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 3\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_100727-dedc0ee5</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/dedc0ee5' target=\"_blank\">quiet-sweep-7</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/dedc0ee5' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/dedc0ee5</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Loss: 2.0547 | Iterations: 40000 | Time: 3267.539\n",
      "Epoch: 2\n",
      "Loss: 1.1102 | Iterations: 40000 | Time: 3300.856\n",
      "Epoch: 3\n",
      "Loss: 0.9200 | Iterations: 40000 | Time: 3186.834\n",
      "Epoch: 4\n",
      "Loss: 0.8456 | Iterations: 40000 | Time: 3524.417\n",
      "Epoch: 5\n",
      "Loss: 0.8036 | Iterations: 40000 | Time: 3374.540\n",
      "Test Accuracy: 0.2593\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3c47487c4fe54ba2897cdb9effdee4a3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.001 MB of 0.006 MB uploaded\\r'), FloatProgress(value=0.18636363636363637, max=1.…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▃▂▁▁</td></tr><tr><td>validation_accuracy</td><td>▁▅▇██</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>5</td></tr><tr><td>test_accuracy</td><td>0.25928</td></tr><tr><td>training_loss</td><td>0.80361</td></tr><tr><td>validation_accuracy</td><td>0.27295</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">quiet-sweep-7</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/dedc0ee5' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/dedc0ee5</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>.\\wandb\\run-20240517_100727-dedc0ee5\\logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: f738j2xs with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: LSTM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 16\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 256\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.005\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.17.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.4"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_165209-f738j2xs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f738j2xs' target=\"_blank\">glowing-sweep-8</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jaevdu4x</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f738j2xs' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/f738j2xs</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Network error (ConnectionError), entering retry loop.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loss: 2.3447 | Iterations: 40000 | Time: 2729.832\n",
      "Epoch: 2\n",
      "Loss: 1.9149 | Iterations: 40000 | Time: 1847.979\n",
      "Epoch: 3\n",
      "Loss: 1.7961 | Iterations: 40000 | Time: 1957.090\n",
      "Epoch: 4\n",
      "Loss: 1.7228 | Iterations: 40000 | Time: 1975.398\n"
     ]
    }
   ],
   "source": [
    "wandb_id = wandb.sweep(sweep_configuration, project=\"Deep Learning Assignment 3\")\n",
    "wandb.agent(wandb_id, train_sweep, count=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "params = {\n",
    "    \"embed_size\": 16,\n",
    "    \"hidden_size\": 512,\n",
    "    \"cell_type\": \"LSTM\",\n",
    "    \"num_layers\": 2,\n",
    "    \"dropout\": 0.1,\n",
    "    \"learning_rate\": 0.005,\n",
    "    \"optimizer\": \"SGD\",\n",
    "    \"teacher_forcing_ratio\": 0.5,\n",
    "    \"max_length\": 50\n",
    "}\n",
    "\n",
    "# model = Translator('tam', params)\n",
    "\n",
    "# epochs = 10\n",
    "\n",
    "# for epoch in range(epochs):\n",
    "#     print(\"Epoch: {}\".format(epoch + 1))\n",
    "\n",
    "#     plot_losses = model.train()\n",
    "\n",
    "#     # take average of plot losses as training loss\n",
    "#     training_loss = sum(plot_losses) / len(plot_losses)\n",
    "\n",
    "#     start_time = time.time()\n",
    "#     training_accuracy = model.test_validate('train')\n",
    "#     print(\"Training Accuracy: {:.4f}\".format(training_accuracy))\n",
    "#     print(\"Time taken to evaluate train: {:.4f}\".format(time.time() - start_time))\n",
    "\n",
    "#     start_time = time.time()\n",
    "#     validation_accuracy = model.test_validate('valid')\n",
    "#     print(\"Validation Accuracy: {:.4f}\".format(validation_accuracy))\n",
    "#     print(\"Time taken to evaluate validation: {:.4f}\".format(time.time() - start_time))\n",
    "\n",
    "# start_time = time.time()\n",
    "# test_accuracy = model.test_validate('test')\n",
    "# print(\"Test Accuracy: {:.4f}\".format(test_accuracy))\n",
    "# print(\"Time taken to evaluate test: {:.4f}\".format(time.time() - start_time))\n",
    "\n",
    "# test_output_file = open(\"test_output.txt\", \"w\")\n",
    "\n",
    "# for pair in get_data('tam', 'test'):\n",
    "#     test_output_file.write(\"{}, {}\\n\".format(pair[0], model.evaluate(pair[0])))\n",
    "\n",
    "# test_output_file.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "params = {\n",
    "    \"embed_size\": 16,\n",
    "    \"hidden_size\": 512,\n",
    "    \"cell_type\": \"LSTM\",\n",
    "    \"num_layers\": 2,\n",
    "    \"dropout\": 0.1,\n",
    "    \"learning_rate\": 0.005,\n",
    "    \"optimizer\": \"SGD\",\n",
    "    \"teacher_forcing_ratio\": 0.5,\n",
    "    \"max_length\": 50\n",
    "}\n",
    "\n",
    "model = Translator('hin', params)\n",
    "model.encoder.load_state_dict(torch.load(\"../best_model_vanilla/encoder.pt\"))\n",
    "model.decoder.load_state_dict(torch.load(\"../best_model_vanilla/decoder.pt\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ஹெல்லோ\n"
     ]
    }
   ],
   "source": [
    "print(model.evaluate(\"hello\"))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
