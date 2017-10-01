import pandas as pd
import os
from settings import *
import numpy as np

def get_more_variables(training=True):
	if training:
		enquiry = pd.read_csv(os.path.join(DATA_DIR, ENQUIRY_TRAINING_FILE))
		account = pd.read_csv(os.path.join(DATA_DIR, ACCOUNT_TRAINING_FILE))
		data = pd.read_csv(os.path.join(DATA_DIR, DATA_TRAINING_FILE))
	else:
		enquiry = pd.read_csv(os.path.join(DATA_DIR, ENQUIRY_TEST_FILE))
		account = pd.read_csv(os.path.join(DATA_DIR, ACCOUNT_TEST_FILE))
		data = pd.read_csv(os.path.join(DATA_DIR, DATA_TEST_FILE))

	#get_ratio_currbalance_creditlimit():
	ratio_currbalance_creditlimit = account.groupby('customer_no')['cur_balance_amt', 'creditlimit'].agg('sum').reset_index()
	ratio_currbalance_creditlimit['ratio_currbalance_creditlimit'] = ratio_currbalance_creditlimit['cur_balance_amt'] / ratio_currbalance_creditlimit['creditlimit']
	ratio_currbalance_creditlimit = ratio_currbalance_creditlimit.drop(['cur_balance_amt', 'creditlimit'], axis = 1)


	#get_max_freq_enquiry():
	enquiry['enq_purpose'] = enquiry['enq_purpose'].astype('category')
	max_freq_enquiry = enquiry[['customer_no','enq_purpose']].groupby(['customer_no']).agg(lambda x:x.value_counts().index[0]).reset_index()


	#get_mean_diff_lastpaymt_opened_dt():
	enquiry['enquiry_dt'] = pd.to_datetime(enquiry['enquiry_dt'])
	enquiry['dt_opened'] = pd.to_datetime(enquiry['dt_opened'])
	enquiry['diff_in_dates'] = (enquiry['dt_opened'] - enquiry['enquiry_dt']).dt.days
	mean_diff_open_enquiry_dt = enquiry.groupby('customer_no')['diff_in_dates'].agg('mean').reset_index(name='mean_diff_open_enquiry_dt')

	#get_count_enquiry_recency_90():
	count_enquiry_recency_90 = enquiry.query('diff_in_dates <= 90').groupby(['customer_no']).size().reset_index(name='#Enquiries90')

	#get_count_enquiry_recency_365():
	count_enquiry_recency_365 = enquiry.query('diff_in_dates <= 365').groupby(['customer_no']).size().reset_index(name='#Enquiries365')

	#get_diff_in_last_pay_date():
	account['opened_dt'] = pd.to_datetime(account['opened_dt'])
	account['last_paymt_dt'] = pd.to_datetime(account['last_paymt_dt'])
	account['diff_in_dates'] = (account['last_paymt_dt'] - account['opened_dt']).dt.days
	diff_in_last_pay_date_sum = account.groupby('customer_no')['diff_in_dates'].agg('sum').reset_index(name='total_diff_lastpaymt_opened_dt')
	diff_in_last_pay_date_mean = account.groupby('customer_no')['diff_in_dates'].agg('mean').reset_index(name='average_diff_lastpaymt_opened_dt')

	#get_payment_history_variable_length():
	account['payment_history_variable_length'] = account['paymenthistory1'].str.len() + account['paymenthistory2'].str.len()
	payment_history_variable_length = account.groupby('customer_no')['payment_history_variable_length'].agg('mean').reset_index(name='payment_history_variable_length')


	def check_secure(row):
		if row['enq_purpose'] in [5, 6, 8, 9, 10, 16, 35, 40, 41, 43, 0, 99]:
		    return 'unsecured'
		elif row['enq_purpose'] == np.nan:
		    return np.nan
		else:
			return 'secured'

	#get_perc_unsecured_others():
	enquiry['security'] = enquiry.apply(check_secure, axis=1)
	total_loan = enquiry.groupby('customer_no')['security'].size().reset_index(name='total_loan')
	secured_load = enquiry[enquiry['security'] == 'secured'].groupby('customer_no')['security'].size().reset_index(name='secured_loan')
	security = total_loan.merge(secured_load, on='customer_no')
	security['perc_unsecured_others'] = security['secured_loan']/ security['total_loan']
	security = security.drop(['secured_loan', 'total_loan'], axis=1)

	#get_utilization_trend():
	total_cur_bal_amt = account.groupby('customer_no')['cur_balance_amt'].agg('sum').reset_index(name='total_cur_bal_amt')
	mean_cur_bal_amt = account.groupby('customer_no')['cur_balance_amt'].agg('mean').reset_index(name='mean_cur_bal_amt')
	mean_credit_limit = account.groupby('customer_no')['creditlimit'].agg('sum').reset_index(name='mean_credit_limit')
	mean_cash_limit = account.groupby('customer_no')['cashlimit'].agg('sum').reset_index(name='mean_cash_limit')
	total_credit_limit = account.groupby('customer_no')['creditlimit'].agg('sum').reset_index(name='total_credit_limit')
	ut = total_cur_bal_amt.    merge(mean_cur_bal_amt,on='customer_no').    merge(mean_credit_limit,how='left',on='customer_no').    merge(mean_cash_limit,how='left',on='customer_no').    merge(total_credit_limit,how='left', on='customer_no')
	    
	ut['utilization_trend'] = (ut['total_cur_bal_amt']/ut['total_credit_limit'])/(ut['mean_cur_bal_amt']/(ut['mean_cash_limit']+ut['mean_credit_limit']))
	ut = ut.drop(['total_cur_bal_amt', 'mean_cur_bal_amt', 'mean_credit_limit', 'mean_cash_limit', 'total_credit_limit'], axis=1)


	#payment_history_avg_dpd_0_29_bucket
	account['dpd_temp'] = account['paymenthistory1'].astype(str).str[3:6]
	account['dpd'] = np.where(account['dpd_temp'].str.lower().str.contains("std|xxx|sub|sma|lss|dbt"), np.nan, account['dpd_temp']).astype('float64')

	payment_history_avg_dpd_0_29_bucket = account[account['dpd'] < 30].groupby('customer_no')['dpd'].size().reset_index(name='payment_history_avg_dpd_0_29_bucket')

	#min_months_last_30_plus
	min_months_last_30_plus = account[account['dpd'] < 30].groupby('customer_no')['dpd'].size().reset_index(name='payment_history_avg_dpd_0_29_bucket')

	final_df = mean_diff_open_enquiry_dt.\
		merge(ratio_currbalance_creditlimit,on='customer_no').\
		merge(count_enquiry_recency_90,how='left',on='customer_no').\
		merge(count_enquiry_recency_365,how='left',on='customer_no').\
		merge(max_freq_enquiry,how='left', on='customer_no').\
		merge(diff_in_last_pay_date_sum,how='left', on='customer_no').\
		merge(diff_in_last_pay_date_mean,how='left', on='customer_no').\
		merge(payment_history_variable_length,how='left', on='customer_no').\
		merge(payment_history_avg_dpd_0_29_bucket,how='left', on='customer_no').\
		merge(ut,how='left', on='customer_no').\
		merge(security,how='left', on='customer_no').\
		merge(min_months_last_30_plus,how='left', on='customer_no')
	if training:
		final_df.to_csv(os.path.join(DATA_DIR, NEW_FILE), sep=',')
	else:
		final_df.to_csv(os.path.join(DATA_DIR, NEW_TEST_FILE), sep=',')

	