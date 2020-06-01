import numpy as np
from read_data import read_data, update_config
from model import BiDAF

class Config(object):
    def __init__(self):
        self.max_num_sents = None
        self.max_sent_size = None
        self.max_ques_size = None
        self.max_word_size = None
        self.max_para_size = None
        self.num_steps = 25000
        self.eval_period = 500
        self.log_period = 100
        self.save_period = 500
        self.val_num_batches = 400
        self.test_num_batches = 0
        self.emb_mat = None
        self.mode = None
        self.num_sents_th = 32
        self.sent_size_th = 500
        self.para_size_th = 260

def _config_debug(config):
    config.num_steps = 2
    config.eval_period = 1
    config.log_period = 1
    config.save_period = 1
    config.val_num_batches = 2
    config.test_num_batches = 2

def train(config):
    train_data = read_data('train')
    dev_data = read_data('dev')

    update_config(config, [train_data, dev_data])
    _config_debug(config)

    word2vec_dict = train_data.shared['lower_word2vec'] 
    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat
    bidaf_model = train_bidaf()
    

def train_bidaf(config, data):
    # train BiDAF model
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model_bidaf = BiDAF(config, train_data)
    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_exact, max_dev_f1 = -1, -1

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch

        p1, p2 = model(batch)

        optimizer.zero_grad()
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_exact, dev_f1 = test(model, ema, args, data)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('exact_match/dev', dev_exact, c)
            writer.add_scalar('f1/dev', dev_f1, c)
            print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
                  f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                max_dev_exact = dev_exact
                best_model_bidaf = copy.deepcopy(model)

            loss = 0
            model.train()

    writer.close()
    print(f'max dev EM: {max_dev_exact:.3f} / max dev F1: {max_dev_f1:.3f}')

    return best_model_bidaf


config = Config()
config.mode = 'train'
train(config)