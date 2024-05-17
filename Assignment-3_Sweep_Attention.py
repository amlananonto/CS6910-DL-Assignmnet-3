{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Currently logged in as: \u001b[33mge22m012\u001b[0m (\u001b[33mge22m012_omlan\u001b[0m). Use \u001b[1m`wandb login --relogin`\u001b[0m to force relogin\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \u001b[33mWARNING\u001b[0m If you're specifying your api key in code, ensure this code is not shared publicly.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \u001b[33mWARNING\u001b[0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: Appending key for api.wandb.ai to your netrc file: C:\\Users\\Amlan\\.netrc\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cuda\n"
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
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
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
   "execution_count": 3,
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
   "cell_type": "code",
   "execution_count": 4,
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
   "execution_count": 5,
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
   "execution_count": 6,
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "class AttentionDecoder(nn.Module):\n",
    "    def __init__(self,\n",
    "                 out_sz: int,\n",
    "                 embed_sz: int,\n",
    "                 hidden_sz: int,\n",
    "                 cell_type: str,\n",
    "                 n_layers: int,\n",
    "                 dropout: float):\n",
    "\n",
    "        super(AttentionDecoder, self).__init__()\n",
    "        self.hidden_sz = hidden_sz\n",
    "        self.n_layers = n_layers\n",
    "        self.dropout = dropout\n",
    "        self.cell_type = cell_type\n",
    "        self.embedding = nn.Embedding(out_sz, embed_sz)\n",
    "\n",
    "        self.attn = nn.Linear(hidden_sz + embed_sz, 50)\n",
    "        self.attn_combine = nn.Linear(hidden_sz + embed_sz, hidden_sz)\n",
    "\n",
    "        self.rnn = get_cell(cell_type)(input_size = hidden_sz,\n",
    "                                       hidden_size = hidden_sz,\n",
    "                                       num_layers = n_layers,\n",
    "                                       dropout = dropout)\n",
    "        \n",
    "        self.out = nn.Linear(hidden_sz, out_sz)\n",
    "        self.softmax = nn.LogSoftmax(dim=1)\n",
    "\n",
    "    def forward(self, input, hidden, cell, encoder_outputs):\n",
    "        embedding = self.embedding(input).view(1, 1, -1)\n",
    "\n",
    "        attn_weights = F.softmax(self.attn(torch.cat((embedding[0], hidden[0]), 1)), dim=1)\n",
    "        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))\n",
    "        \n",
    "        output = torch.cat((embedding[0], attn_applied[0]), 1)\n",
    "        output = self.attn_combine(output).unsqueeze(0)\n",
    "\n",
    "        if(self.cell_type == \"LSTM\"):\n",
    "            output, (hidden, cell) = self.rnn(output, (hidden, cell))\n",
    "        else:\n",
    "            output, hidden = self.rnn(output, hidden)\n",
    "\n",
    "        output = self.softmax(self.out(output[0]))\n",
    "        return output, hidden, cell, attn_weights\n",
    "    \n",
    "    def initHidden(self):\n",
    "        return torch.zeros(self.n_layers, 1, self.hidden_sz, device=device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
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
   "execution_count": 9,
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
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "PRINT_EVERY = 40000\n",
    "PLOT_EVERY = 40000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
    "        self.decoder = AttentionDecoder(out_sz = self.output_size,\n",
    "                             embed_sz = params[\"embed_size\"],\n",
    "                             hidden_sz = params[\"hidden_size\"],\n",
    "                             cell_type = params[\"cell_type\"],\n",
    "                             n_layers = params[\"num_layers\"],\n",
    "                             dropout = params[\"dropout\"]).to(device)\n",
    "\n",
    "        self.encoder_optimizer = get_optimizer(params[\"optimizer\"])(self.encoder.parameters(), lr=params[\"learning_rate\"], weight_decay=params[\"weight_decay\"])\n",
    "        self.decoder_optimizer = get_optimizer(params[\"optimizer\"])(self.decoder.parameters(), lr=params[\"learning_rate\"], weight_decay=params[\"weight_decay\"])\n",
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
    "                decoder_output, decoder_hidden, decoder_cell, decoder_attention = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)\n",
    "                loss += self.criterion(decoder_output, target_tensor[di])\n",
    "\n",
    "                decoder_input = target_tensor[di]\n",
    "        else:\n",
    "            for di in range(target_length):\n",
    "                decoder_output, decoder_hidden, decoder_cell, decoder_attention = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)\n",
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
    "        for iter in range(1, iters):\n",
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
    "            decoder_attentions = torch.zeros(self.max_length, self.max_length)\n",
    "\n",
    "            for di in range(self.max_length):\n",
    "                decoder_output, decoder_hidden, decoder_cell, decoder_attention = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)\n",
    "                decoder_attentions[di] = decoder_attention.data\n",
    "                topv, topi = decoder_output.topk(1)\n",
    "                \n",
    "                if topi.item() == EOS_token:\n",
    "                    break\n",
    "                else:\n",
    "                    decoded_chars += self.output_lang.index2word[topi.item()]\n",
    "\n",
    "                decoder_input = topi.squeeze().detach()\n",
    "\n",
    "            return decoded_chars, decoder_attentions[:di + 1]\n",
    "        \n",
    "    def test_validate(self, type:str):\n",
    "        pairs = get_data(self.lang, type)\n",
    "        accuracy = 0\n",
    "        for pair in pairs:\n",
    "            output, _ = self.evaluate(pair[0])\n",
    "            if output == pair[1]:\n",
    "                accuracy += 1\n",
    "        return accuracy / len(pairs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
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
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "sweep_configuration = {\n",
    "    \"method\": \"bayes\",\n",
    "    \"name\" : \"valo_sweep_att\",\n",
    "    \"metric\": {\n",
    "        \"name\": \"validation_accuracy\",\n",
    "        \"goal\": \"maximize\"\n",
    "    },\n",
    "    \"parameters\": {\n",
    "        \"embed_size\": {\n",
    "            \"values\": [8, 16, 32]\n",
    "        },\n",
    "        \"hidden_size\": {\n",
    "            \"values\": [64, 128, 256, 512]\n",
    "        },\n",
    "        \"cell_type\": {\n",
    "            \"values\": [\"RNN\", \"LSTM\", \"GRU\"]\n",
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
    "        },\n",
    "        \"weight_decay\": {\n",
    "            \"values\": [0, 1e-1, 1e-3, 1e-5]\n",
    "        }\n",
    "    }\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
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
    "    run.name = \"embed_size: {} | hidden_size: {} | cell_type: {} | num_layers: {} | dropout: {} | learning_rate: {} | optimizer: {} | teacher_forcing_ratio: {} | max_length: {} | weight_decay: {}\".format(config.embed_size, config.hidden_size, config.cell_type, config.num_layers, config.dropout, config.learning_rate, config.optimizer, config.teacher_forcing_ratio, config.max_length, config.weight_decay)\n",
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
    "            if validation_accuracy < 0.9 * old_validation_accuracy:\n",
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
      "Create sweep with ID: jzw8nie9\n",
      "Sweep URL: https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: a95bksf3 with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: GRU\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 32\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 128\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.0005\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 3\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: SGD\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tweight_decay: 0\n"
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
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_013535-a95bksf3</code>"
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
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/a95bksf3' target=\"_blank\">driven-sweep-1</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/a95bksf3' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/a95bksf3</a>"
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
      "Loss: 2.9390 | Iterations: 40000 | Time: 2290.471\n",
      "Epoch: 2\n",
      "Loss: 2.8795 | Iterations: 40000 | Time: 2380.158\n",
      "Epoch: 3\n",
      "Loss: 2.5434 | Iterations: 40000 | Time: 2390.889\n",
      "Epoch: 4\n",
      "Loss: 2.1065 | Iterations: 40000 | Time: 2288.215\n",
      "Epoch: 5\n",
      "Loss: 1.7530 | Iterations: 40000 | Time: 2374.407\n",
      "Test Accuracy: 0.0352\n"
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
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>██▆▃▁</td></tr><tr><td>validation_accuracy</td><td>▁▁▁▄█</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>5</td></tr><tr><td>test_accuracy</td><td>0.03516</td></tr><tr><td>training_loss</td><td>1.75296</td></tr><tr><td>validation_accuracy</td><td>0.04443</td></tr></table><br/></div></div>"
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
       " View run <strong style=\"color:#cdcd00\">driven-sweep-1</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/a95bksf3' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/a95bksf3</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
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
       "Find logs at: <code>.\\wandb\\run-20240517_013535-a95bksf3\\logs</code>"
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
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: 6leo21f6 with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: GRU\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 8\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 256\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.0005\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 1\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tweight_decay: 0.001\n"
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
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_055310-6leo21f6</code>"
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
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6leo21f6' target=\"_blank\">young-sweep-2</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6leo21f6' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6leo21f6</a>"
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
      "Loss: 1.3489 | Iterations: 40000 | Time: 2177.174\n",
      "Epoch: 2\n",
      "Loss: 0.9520 | Iterations: 40000 | Time: 2204.696\n",
      "Epoch: 3\n",
      "Loss: 0.8689 | Iterations: 40000 | Time: 2215.094\n",
      "Epoch: 4\n",
      "Loss: 0.8268 | Iterations: 40000 | Time: 2217.546\n",
      "Epoch: 5\n",
      "Loss: 0.7992 | Iterations: 40000 | Time: 2128.973\n",
      "Test Accuracy: 0.2666\n"
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
       "VBox(children=(Label(value='0.001 MB of 0.006 MB uploaded\\r'), FloatProgress(value=0.1942200124300808, max=1.0…"
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
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▃▂▁▁</td></tr><tr><td>validation_accuracy</td><td>▁▅▆▆█</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>5</td></tr><tr><td>test_accuracy</td><td>0.2666</td></tr><tr><td>training_loss</td><td>0.79923</td></tr><tr><td>validation_accuracy</td><td>0.27051</td></tr></table><br/></div></div>"
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
       " View run <strong style=\"color:#cdcd00\">young-sweep-2</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6leo21f6' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6leo21f6</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
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
       "Find logs at: <code>.\\wandb\\run-20240517_055310-6leo21f6\\logs</code>"
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
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: 8frnifce with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: LSTM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0.1\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 32\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 128\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.005\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tweight_decay: 0.1\n"
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
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_095338-8frnifce</code>"
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
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/8frnifce' target=\"_blank\">dry-sweep-3</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/8frnifce' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/8frnifce</a>"
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
      "Loss: 3.0502 | Iterations: 40000 | Time: 2178.350\n",
      "Epoch: 2\n",
      "Loss: 3.0789 | Iterations: 40000 | Time: 2325.269\n",
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
       "VBox(children=(Label(value='0.001 MB of 0.006 MB uploaded\\r'), FloatProgress(value=0.19334880123743234, max=1.…"
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
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>▁█</td></tr><tr><td>validation_accuracy</td><td>▁▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>2</td></tr><tr><td>test_accuracy</td><td>0.0</td></tr><tr><td>training_loss</td><td>3.07886</td></tr><tr><td>validation_accuracy</td><td>0.0</td></tr></table><br/></div></div>"
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
       " View run <strong style=\"color:#cdcd00\">dry-sweep-3</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/8frnifce' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/8frnifce</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
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
       "Find logs at: <code>.\\wandb\\run-20240517_095338-8frnifce\\logs</code>"
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
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: r2mkapve with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: GRU\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 16\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 256\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.001\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tweight_decay: 0.001\n"
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
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_113423-r2mkapve</code>"
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
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/r2mkapve' target=\"_blank\">eager-sweep-4</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/r2mkapve' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/r2mkapve</a>"
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
      "Loss: 1.3722 | Iterations: 40000 | Time: 3136.008\n",
      "Epoch: 2\n",
      "Loss: 1.0275 | Iterations: 40000 | Time: 3048.591\n",
      "Epoch: 3\n"
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
      "Loss: 0.9723 | Iterations: 40000 | Time: 5707.243\n",
      "Epoch: 4\n",
      "Loss: 0.9448 | Iterations: 40000 | Time: 3151.034\n",
      "Epoch: 5\n"
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
      "Loss: 0.9303 | Iterations: 40000 | Time: 3560.240\n",
      "Test Accuracy: 0.2117\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3f6becb848964c27b111ad6b7bcda16b",
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
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>▁▃▅▆█</td></tr><tr><td>test_accuracy</td><td>▁</td></tr><tr><td>training_loss</td><td>█▃▂▁▁</td></tr><tr><td>validation_accuracy</td><td>▁▄▅██</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>epoch</td><td>5</td></tr><tr><td>test_accuracy</td><td>0.21167</td></tr><tr><td>training_loss</td><td>0.93026</td></tr><tr><td>validation_accuracy</td><td>0.229</td></tr></table><br/></div></div>"
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
       " View run <strong style=\"color:#cdcd00\">eager-sweep-4</strong> at: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/r2mkapve' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/r2mkapve</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
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
       "Find logs at: <code>.\\wandb\\run-20240517_113423-r2mkapve\\logs</code>"
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
      "\u001b[34m\u001b[1mwandb\u001b[0m: Agent Starting Run: 6y080x0n with config:\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tcell_type: GRU\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tdropout: 0\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tembed_size: 8\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \thidden_size: 512\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tlearning_rate: 0.0005\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tmax_length: 50\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tnum_layers: 2\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \toptimizer: ADAM\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tteacher_forcing_ratio: 0.5\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: \tweight_decay: 0.001\n"
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
       "Run data is saved locally in <code>D:\\assignment 3\\wandb\\run-20240517_180409-6y080x0n</code>"
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
       "Syncing run <strong><a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6y080x0n' target=\"_blank\">firm-sweep-5</a></strong> to <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>Sweep page: <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View sweep at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/sweeps/jzw8nie9</a>"
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
       " View run at <a href='https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6y080x0n' target=\"_blank\">https://wandb.ai/ge22m012_omlan/Deep%20Learning%20Assignment%203/runs/6y080x0n</a>"
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
      "Loss: 1.4627 | Iterations: 40000 | Time: 2849.560\n",
      "Epoch: 2\n"
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
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: '../best_models_attn/encoder.pt'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn [15], line 16\u001b[0m\n\u001b[0;32m      1\u001b[0m params \u001b[38;5;241m=\u001b[39m {\n\u001b[0;32m      2\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124membed_size\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m32\u001b[39m,\n\u001b[0;32m      3\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhidden_size\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m256\u001b[39m,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     11\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mweight_decay\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m0.001\u001b[39m\n\u001b[0;32m     12\u001b[0m }\n\u001b[0;32m     14\u001b[0m model \u001b[38;5;241m=\u001b[39m Translator(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhin\u001b[39m\u001b[38;5;124m\"\u001b[39m, params)\n\u001b[1;32m---> 16\u001b[0m model\u001b[38;5;241m.\u001b[39mencoder\u001b[38;5;241m.\u001b[39mload_state_dict(\u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43m../best_models_attn/encoder.pt\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m)\n\u001b[0;32m     17\u001b[0m model\u001b[38;5;241m.\u001b[39mdecoder\u001b[38;5;241m.\u001b[39mload_state_dict(torch\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m../best_models_attn/decoder.pt\u001b[39m\u001b[38;5;124m'\u001b[39m))\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\torch\\serialization.py:791\u001b[0m, in \u001b[0;36mload\u001b[1;34m(f, map_location, pickle_module, weights_only, **pickle_load_args)\u001b[0m\n\u001b[0;32m    788\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mencoding\u001b[39m\u001b[38;5;124m'\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m pickle_load_args\u001b[38;5;241m.\u001b[39mkeys():\n\u001b[0;32m    789\u001b[0m     pickle_load_args[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mencoding\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mutf-8\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[1;32m--> 791\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[43m_open_file_like\u001b[49m\u001b[43m(\u001b[49m\u001b[43mf\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mrb\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m opened_file:\n\u001b[0;32m    792\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m _is_zipfile(opened_file):\n\u001b[0;32m    793\u001b[0m         \u001b[38;5;66;03m# The zipfile reader is going to advance the current file position.\u001b[39;00m\n\u001b[0;32m    794\u001b[0m         \u001b[38;5;66;03m# If we want to actually tail call to torch.jit.load, we need to\u001b[39;00m\n\u001b[0;32m    795\u001b[0m         \u001b[38;5;66;03m# reset back to the original position.\u001b[39;00m\n\u001b[0;32m    796\u001b[0m         orig_position \u001b[38;5;241m=\u001b[39m opened_file\u001b[38;5;241m.\u001b[39mtell()\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\torch\\serialization.py:271\u001b[0m, in \u001b[0;36m_open_file_like\u001b[1;34m(name_or_buffer, mode)\u001b[0m\n\u001b[0;32m    269\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_open_file_like\u001b[39m(name_or_buffer, mode):\n\u001b[0;32m    270\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m _is_path(name_or_buffer):\n\u001b[1;32m--> 271\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43m_open_file\u001b[49m\u001b[43m(\u001b[49m\u001b[43mname_or_buffer\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmode\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    272\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    273\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mw\u001b[39m\u001b[38;5;124m'\u001b[39m \u001b[38;5;129;01min\u001b[39;00m mode:\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\torch\\serialization.py:252\u001b[0m, in \u001b[0;36m_open_file.__init__\u001b[1;34m(self, name, mode)\u001b[0m\n\u001b[0;32m    251\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, name, mode):\n\u001b[1;32m--> 252\u001b[0m     \u001b[38;5;28msuper\u001b[39m()\u001b[38;5;241m.\u001b[39m\u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mname\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmode\u001b[49m\u001b[43m)\u001b[49m)\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '../best_models_attn/encoder.pt'"
     ]
    }
   ],
   "source": [
    "params = {\n",
    "    \"embed_size\": 32,\n",
    "    \"hidden_size\": 256,\n",
    "    \"cell_type\": \"RNN\",\n",
    "    \"num_layers\": 2,\n",
    "    \"dropout\": .1,\n",
    "    \"learning_rate\": 0.001,\n",
    "    \"optimizer\": \"SGD\",\n",
    "    \"teacher_forcing_ratio\": 0.5,\n",
    "    \"max_length\": 50,\n",
    "    \"weight_decay\": 0.001\n",
    "}\n",
    "\n",
    "model = Translator(\"hin\", params)\n",
    "\n",
    "model.encoder.load_state_dict(torch.load('../best_models_attn/encoder.pt'))\n",
    "model.decoder.load_state_dict(torch.load('../best_models_attn/decoder.pt'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_data = get_data(\"tam\", \"test\") \n",
    "test_pairs_10  = random.sample(test_data, 10)\n",
    "print(test_pairs_10)\n",
    "\n",
    "for i in range(9):\n",
    "    pair = test_pairs_10[i]\n",
    "    output, attentions = model.evaluate(pair[0])\n",
    "    print(\"Input: {}, Target: {}, Output: {}\".format(pair[0], pair[1], output))\n",
    "    \n",
    "    attentions = attentions[:, :(10 + len(output))]\n",
    "    plt.matshow(attentions.numpy())\n",
    "    plt.show()\n",
    "\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
