import datetime, os

class Args:
    def __init__(self, epochs = 20, batch_size=64, 
                 AL_iters = 10, AL_batch=32, AL_select='acc',
                 init_num = 100, val_ratio =0.2, 
                 problem_type = 'binary', 
                 model_type='NN', model_args={'n_hidden':32, 'p_dropout':0.2},
                 dataset = 'german', save_model=True, save_dir = None):
        
        self.set_train_params(epochs, batch_size, init_num, val_ratio)
        self.set_AL_params(AL_iters,AL_batch,AL_select)
        self.set_problem_params(problem_type, model_type, model_args)
        self.dataset = dataset
        self.save_model = save_model
        print("need to set selection params!")
        if save_model:
            self.make_save_dir(save_dir)
            print("save directory: ", self.save_dir)
    
    def set_train_params(self, epochs, batch_size, init_num, val_ratio=0.2):
        self.epochs = epochs # epochs for training
        self.batch_size = batch_size # batch size for training
        self.init_num = init_num # initial number for the first training
        self.val_ratio = val_ratio # valication set ratio for selection (train_AL_valid)
        
    def set_AL_params(self, AL_iters, AL_batch, AL_select):
        self.AL_iters = AL_iters # AL batch 몇 번 뽑는지?
        self.AL_batch = AL_batch # AL 시에 select 되는 데이터 수
        self.AL_select = AL_select # AL 시에 criterion ['acc', 'loss']
        
    def set_problem_params(self, problem_type, model_type, model_args):
        self.problem_type = problem_type
        self.model_type = model_type
        self.model_args = model_args
        
    def set_selection_params(self, sel_fn, sel_args):    
        self.sel_fn = sel_fn
        self.sel_args = sel_args
    
    def make_save_dir(self, save_dir):
        if save_dir is None:
            save_dir = "../results/"+self.dataset
        
        time_dir = datetime.datetime.now().strftime('%Y%m%d_%H%M')[2:]
        self.save_dir = os.path.join(save_dir, time_dir)
        os.makedirs(self.save_dir)
        
    def print_args(self):
        print("train epochs/batch: {}/{}".format(self.epochs,self.batch_size))
        print("AL iters/batch: {}/{}".format(self.AL_iters,self.AL_batch))
        print("AL selection is based on ", self.AL_select)
        print("initial number and validation ratio are {}/{}".format(self.init_num,self.val_ratio))
        print("problem and models are {}/{}".format(self.problem_type, self.model_type))