#!/usr/bin/env python2

from lib import *
from sampler import *

from agents import *
from emulator import *
from simulators import *
from visualizer import *
from company_data_sampler import CompanyDataSampler

# import cProfile, pstats, io


def get_model(model_type, env, learning_rate, fld_load):

	print_t = False
	exploration_init = 1.

	if model_type == 'MLP':
		m = 16
		layers = 5
		hidden_size = [m]*layers
		model = QModelMLP(env.state_shape, env.n_action)
		model.build_model(hidden_size, learning_rate=learning_rate, activation='tanh')

	elif model_type == 'conv':

		m = 16
		layers = 2
		filter_num = [m]*layers
		filter_size = [3] * len(filter_num)
		#use_pool = [False, True, False, True]
		#use_pool = [False, False, True, False, False, True]
		use_pool = None
		#dilation = [1,2,4,8]
		dilation = None
		dense_units = [48,24]
		model = QModelConv(env.state_shape, env.n_action)
		model.build_model(filter_num, filter_size, dense_units, learning_rate,
			dilation=dilation, use_pool=use_pool)

	elif model_type == 'RNN-LSTM':
		m = 32
		layers = 3
		hidden_size = [m]*layers
		dense_units = [m,m]
		model = QModelLSTM(env.state_shape, env.n_action)
		model.build_model(hidden_size, dense_units, learning_rate=learning_rate)
		print_t = True

	elif model_type == 'RNN-GRU':
		m = 32
		layers = 3
		hidden_size = [m]*layers
		dense_units = [m,m]
		model = QModelGRU(env.state_shape, env.n_action)
		model.build_model(hidden_size, dense_units, learning_rate=learning_rate)
		print_t = True

	elif model_type == 'ConvRNN':

		m = 8
		conv_n_hidden = [m,m]
		RNN_n_hidden = [m,m]
		dense_units = [m,m]
		model = QModelConvGRU(env.state_shape, env.n_action)
		model.build_model(conv_n_hidden, RNN_n_hidden, dense_units, learning_rate=learning_rate)
		print_t = True

	elif model_type == 'pretrained':
		model = load_model(fld_load, learning_rate)

	else:
		raise ValueError

	return model, print_t


def main_run(paths):

	"""
	it is recommended to generate database usng sampler.py before run main
	"""

	model_type = 'pretrained'; exploration_init = 1.
	fld_load = paths['model-root-path']
	n_episode_training = 15
	n_episode_testing = 5
	open_cost = 3.3
	#db_type = 'SinSamplerDB'; db = 'concat_half_base_'; Sampler = SinSampler
	db_type = 'PairSamplerDB'; db = 'randjump_100,1(10, 30)[]_'
	batch_size = 8
	learning_rate = 1e-4
	discount_factor = 0.8
	exploration_decay = 0.99
	exploration_min = 0.01
	window_state = 14

	fld = os.path.join('..','data',db_type,db+'A')
	# sampler = Sampler('load', fld=fld)
	sampler = CompanyDataSampler(company_code=paths['company'])
	env = Market(sampler, window_state, open_cost, paths['return-data'])

	model, print_t = get_model(model_type, env, learning_rate, fld_load)
	model.model.summary()
	#return

	agent = Agent(model, discount_factor=discount_factor, batch_size=batch_size)
	visualizer = Visualizer(env.action_labels)

	fld_save = os.path.join(OUTPUT_FLD, sampler.title, model.model_name,
		str((env.window_state, sampler.window_episode, agent.batch_size, learning_rate,
			agent.discount_factor, exploration_decay, env.open_cost)))

	print('='*20)
	print(fld_save)
	print('='*20)

	simulator = Simulator(agent, env, visualizer=visualizer, fld_save=fld_save)
	# simulator.train(n_episode_training, save_per_episode=1, exploration_decay=exploration_decay,
	# 	exploration_min=exploration_min, print_t=print_t, exploration_init=exploration_init)
	#agent.model = load_model(os.path.join(fld_save,'model'), learning_rate)

	#print('='*20+'\nin-sample testing\n'+'='*20)
	simulator.test(1, save_per_episode=1, subfld='in-sample testing')

	"""
	fld = os.path.join('data',db_type,db+'B')
	sampler = SinSampler('load',fld=fld)
	simulator.env.sampler = sampler
	simulator.test(n_episode_testing, save_per_episode=1, subfld='out-of-sample testing')
	"""

def main():

	root_path = "D:\MSC-Project\company-data\P0717"
	companies = [
		'AEL', 'HHL','JKH', 'REXP', 'SAMP',  'VONE' ,'TKYO',
	]

	# paths = []

	for company in companies:
		paths = {}
		paths['return-data'] = os.path.join(root_path,'CompanyData-'+company,'records.csv')
		paths['data-path'] = os.path.join(root_path,'CompanyData-'+company, company + '.csv')
		paths['company'] = company

		if (os.path.exists(os.path.join(root_path,'CompanyData-'+company,'GRU'))):
			paths['return-data'] = os.path.join(root_path,'CompanyData-'+company,'gru-records.csv')
			paths['model-root-path'] = os.path.join(root_path,'CompanyData-'+company,'GRU','model' )
			main_run(paths)

		if (os.path.exists(os.path.join(root_path,'CompanyData-'+company,'LSTM'))):
			paths['return-data'] = os.path.join(root_path,'CompanyData-'+company,'lstm-records.csv')
			paths['model-root-path'] = os.path.join(root_path,'CompanyData-'+company,'LSTM','model' )
			main_run(paths)


if __name__ == '__main__':
	main()
	# profile()
