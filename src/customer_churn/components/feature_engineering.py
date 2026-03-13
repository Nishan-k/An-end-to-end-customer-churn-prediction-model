import pandas as pd
import os
import sys
from src.customer_churn.logging.logger import logging
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.entity.artifact_entity import DataCleaningArtifacts, FeatureEngineeringArtifacts
from src.customer_churn.config.configuration import FeatureEngineeringConfig
import time



class FeatureEngineering:

    def __init__(self, feature_engineering_config: FeatureEngineeringConfig, 
                 data_cleaning_artifacts:DataCleaningArtifacts,
                churn_cutoff_date: str):
        try:
            self.churn_cutoff_date = pd.to_datetime(churn_cutoff_date)
            self.observation_date = self.churn_cutoff_date - pd.DateOffset(months=3)
            self.feature_engineering_config = feature_engineering_config
            self.data_cleaning_artifacts = data_cleaning_artifacts
            self.clipped_train_file_path = self.data_cleaning_artifacts.clipped_training_file_path
            self.clipped_test_file_path = self.data_cleaning_artifacts.clipped_testing_file_path
            self.unclipped_train_file_path = self.data_cleaning_artifacts.unclipped_training_file_path
            self.unclipped_test_file_path = self.data_cleaning_artifacts.unclipped_testing_file_path
            
            logging.info(f"FeatureEngineering Class initialization success!")
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    def create_observation_df(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Filter transactions to those occurring within the observation window.
        The observation window is defined as all transactions with invoicedate <= self.observation_date.
        This data is used to build customer features, ensuring that future churn window transactions are excluded.

        Parameters
        ----------
        df : pd.DataFrame -> Raw transaction data (from data cleaning artifacts).

        Returns
        -------
        pd.DataFrame ->  DataFrame containing only observation window transactions.

        """
        try:
            logging.info("Creating observation dataframe:")
            observation_df = df[df['invoicedate'] <= self.observation_date].copy()
            min_date, max_date = observation_df['invoicedate'].min(), observation_df['invoicedate'].max()
            logging.info(f"The shape of the observation dataframe is: {observation_df.shape}.")
            logging.info(f"The transactiosn in the observation window ranges between: {min_date} : {max_date}")
            logging.info("Creation of observation dataframe success!")
            return observation_df
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def create_stockcode_lvl_metrics(self, observation_df:pd.DataFrame) -> pd.DataFrame:
        """
        Compute product-level metrics: total orders per product and issue rate.
        For each stock code, counts how many orders included that product and how many of those orders
        had an issue (cancellation/return). Then calculates the issue rate per product.
        
        (Important) -> This should be called only on `training observation data` to prevent data leakage.

        Parameters
        ----------
        observation_df : pd.DataFrame -> Observation window transactions (output of create_observation_df).

        Returns
        -------
        pd.DataFrame -> product/stockcode level metrics
        """
        try:
            logging.info("Creating the product/stockcode level metrics:")
            stock_code_df= observation_df[['invoice', 'stockcode', 'quantity', 'order_issue']].copy()
            logging.info(f"There are a total of {stock_code_df['stockcode'].nunique():,} unique products.")
            
            # Adding total_n_orders and total_n_order_issues per stockcode:
            stock_code_df_aggregated = stock_code_df.groupby('stockcode').agg(
                total_n_orders_of_prod = ('invoice', 'nunique'),
                total_n_order_issues_with_prod = ('order_issue', 'sum')).reset_index()

            # Adding total_order_issue_rate:
            stock_code_df_aggregated['total_order_issue_rate_w_prod'] = (stock_code_df_aggregated['total_n_order_issues_with_prod'] / 
            stock_code_df_aggregated['total_n_orders_of_prod'])
            logging.info("Success for creating the product/stockcode level metrics.")
            return stock_code_df_aggregated                
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def create_customer_risk_at_product_lvl(self, observation_df:pd.DataFrame, 
                                            stock_code_df_aggregated:pd.DataFrame) -> pd.DataFrame:
        """
        For each customer, summarize the riskiness of the products they bought.

        Uses the product-level metrics (from create_stockcode_lvl_metrics) and, for each customer,
        calculates the average and maximum issue rate of the products they purchased, as well as the
        total number of orders those products have received across all customers (represents product popularity).

        Parameters
        ----------
        observation_df : pd.DataFrame -> Observation window transactions.
        stock_code_df_aggregated : pd.DataFrame -> Product-level metrics (output of create_stockcode_lvl_metrics).

        Returns
        -------
        pd.DataFrame -> customers and the product/stockcode metrics that they bought representing the 
        risk level of the products they bought.
        """
        try:
            logging.info("Creating the customer level risk for the products they buy.")
            product_per_customer = observation_df[['customerid', 'stockcode']].drop_duplicates()

            # Join the stock level metrics with the product per customer df:
            customer_and_prod_lvl_metrics = stock_code_df_aggregated.merge(right=product_per_customer,
                                                           how='right',
                                                          on='stockcode')
            # Aggregate by customer and generate the customer order metrics/risk at product level:
            customer_risk_at_product_lvl = customer_and_prod_lvl_metrics.groupby(['customerid']).agg(
                                            avg_stockcode_issue_rate=('total_order_issue_rate_w_prod', 'mean'),
                                            max_stockcode_issue_rate=('total_order_issue_rate_w_prod', 'max'),
                                            total_orders_made_for_stock = ('total_n_orders_of_prod', 'sum')
                                        ).reset_index()
            logging.info("Success for creating the customer level risk for the products they buy.")
            return customer_risk_at_product_lvl                
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def create_customer_lvl_agg_metrics(self, observation_df:pd.DataFrame, 
                                        customer_risk_at_product_lvl:pd.DataFrame,
                                       use_clipped:bool) -> pd.DataFrame:
        """
        Build final customer-level features from transaction history and product risk.


        Parameters
        ----------
        observation_df : pd.DataFrame -> Observation window transactions.
        customer_risk_at_product_lvl : pd.DataFrame -> Customer-level product risk table 
                                    (output of create_customer_risk_at_product_lvl).
        use_clipped : bool (If True, uses clipped columns (for linear models); 
        otherwise uses raw columns (for tree models).

        Returns
        -------
        pd.DataFrame
            Final customer-level features, ready for labeling the churn.
        """
        try:
            logging.info(f"Creating the customer level aggregated metrics. Using clipped data: {use_clipped}")
            if use_clipped:
                qty_col = 'quantity_abs_clipped'
                price_col = 'price_clipped'
            else:
                qty_col = 'quantity_abs'
                price_col = 'price'
                
                
            customer_data = observation_df[['invoice', 'quantity', 'invoicedate', 'customerid', 'country', 'order_issue',
                           qty_col, price_col]].copy()
            # Create customer level features:
            customer_data['total_amt_spent'] = customer_data['quantity'] * customer_data[price_col]
            customer_level_metrics = customer_data.groupby(by='customerid').agg(
                                    # 1. Frequency: Total num of unique orders made per customer:
                                    order_frequency = ('invoice', 'nunique'),
                                    # 2. Monetary: Total amount spent by each customer:
                                    total_monetary_value = ('total_amt_spent', 'sum'),
                                    # 3. Total quantity(abs value): involves purchased and had intentions to acquire:
                                    total_quantity_abs = (qty_col, 'sum'),
                                    # 4. Total issues in the orders made (order cancellation or products returned):
                                    total_order_issues = ('order_issue', 'sum'),
                                    # 5. First and last purchase made:
                                    first_purchase = ('invoicedate', 'min'),
                                    last_purchase = ('invoicedate', 'max'),
                                    # 6. Country: most probably has one country per customer but still a condition is defined:
                                    country = ('country', lambda c: c.mode()[0] if not c.mode().empty else 'Unspecified')
                                ).reset_index()
            customer_level_metrics['avg_order_value'] = (customer_level_metrics['total_monetary_value'] 
                                                         / customer_level_metrics['order_frequency'])
            customer_level_metrics['avg_quantity_per_order'] = (customer_level_metrics['total_quantity_abs'] 
                                                                / customer_level_metrics['order_frequency'])
            customer_level_metrics['customer_order_issue_rate'] = (customer_level_metrics['total_order_issues'] 
                                                                   / customer_level_metrics['order_frequency'])
            customer_level_metrics['recency'] = (self.churn_cutoff_date - customer_level_metrics['last_purchase']).dt.days
            customer_level_metrics['tenure'] = (self.churn_cutoff_date - customer_level_metrics['first_purchase']).dt.days

            # Dropping the columns that are not needed anymore:
            customer_level_metrics.drop(columns=['first_purchase', 'last_purchase'], inplace=True)

            # Merging two dfs to create the final customer level features: df for stockcode level metrics and customer level metrics:
            final_customer_lvl_features = customer_level_metrics.merge(customer_risk_at_product_lvl,
                                             on='customerid',
                                             how='left')
            logging.info("Success in creating the customer level aggregated metrics.")
            return final_customer_lvl_features               
        except Exception as e:
            raise CustomerChurnException(e, sys)
    

    def define_churn_label(self, df:pd.DataFrame, final_customer_lvl_features:pd.DataFrame) -> pd.DataFrame:
        """
        Takes in the final_customer_lvl_features df and add churn label to the 
        customer feature table(1=churn, 0=retained).

        Identifies customers who made at least one purchase in the churn window
        (observation_date < invoicedate <= churn_cutoff_date) and labels them as retained (0);
        all others are churned (1). The churn window data is taken from the original DataFrame `df`.

        Parameters
        ----------
        df : pd.DataFrame -> Original transaction data used to derive churn df using self.churn_cutoff_date
        final_customer_lvl_features : pd.DataFrame -> Customer feature table (from create_customer_lvl_agg_metrics).

        Returns
        -------
        pd.DataFrame ->  final_customer_lvl_features with an extra feature 'churn'
        """
        try:
            logging.info("Defining the churn label:")
            logging.info(f"The observation window will be until: {self.observation_date}.")
            logging.info(f"The churn window will be from: {self.observation_date + pd.DateOffset(days=1)} until:   {self.churn_cutoff_date}")
            churn_df = df[(df['invoicedate'] > self.observation_date) & (df['invoicedate'] <= self.churn_cutoff_date)].copy()
            logging.info(f"The min invoice date in churn df: {churn_df['invoicedate'].min()} and the max is: {churn_df['invoicedate'].max()}")
            retained_customer_ids = set(churn_df['customerid'].unique())
            logging.info(f"There are a total of: {len(retained_customer_ids)} customer IDs who made the transaction in churn window.")
            final_customer_lvl_features['churn'] = (~final_customer_lvl_features['customerid'].isin(retained_customer_ids)).astype(int)
            logging.info(f"The count for the churn label is:\n{final_customer_lvl_features['churn'].value_counts()}")
            logging.info(f"The shape of the finalized df is:{final_customer_lvl_features.shape}")
            logging.info("Success at defining the churn label.")
            return final_customer_lvl_features                   
        except Exception as e:
            raise CustomerChurnException(e, sys)
    

    def initiate_feature_engineering(self, branch_name:str):
        """
        Initiates the full feature engineering pipeline for a given data branch.

        Parameters:
        -----------
        branch_name: clipped or unclipped
        clipped: data clipped at 99th quantile to clip the outlier values for the linear models
        unclipped: original data for tree models
        """
        
        starting_time = time.perf_counter()
        logging.info(f"Feature engineering pipeline started, given branch name: {branch_name}")
        use_clipped = False
        try:
            
            train_attr = f"{branch_name}_train_file_path"
            test_attr = f"{branch_name}_test_file_path"

            train_path = getattr(self, train_attr)
            test_path = getattr(self, test_attr)

            # Load the data:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            logging.info(f"Reading {branch_name} training and testing data completed.")
            logging.info(f"Training {branch_name} data has a shape of: {train_df.shape}")
            logging.info(f"Testing {branch_name} data has a shape of: {test_df.shape}")
            
            use_clipped = (branch_name == 'clipped')

            # 1. Create the observation records:
            train_df_obs = self.create_observation_df(df=train_df)
            test_df_obs = self.create_observation_df(df=test_df)

            # 2. Create the stock-code level metrics only using training data to avoid data leakage:
            train_stock_code_metrics_df = self.create_stockcode_lvl_metrics(observation_df=train_df_obs)
            

            # 3. Create customer risk at product/stock-code level:
            train_customer_risk_at_product_lvl = self.create_customer_risk_at_product_lvl(observation_df=train_df_obs,                                                                                    stock_code_df_aggregated=train_stock_code_metrics_df)
            test_customer_risk_at_product_lvl = self.create_customer_risk_at_product_lvl(observation_df=test_df_obs,                                                                                  stock_code_df_aggregated=train_stock_code_metrics_df)

            # 4. Create customer level aggregated metrics:
            train_final_customer_lvl_features = self.create_customer_lvl_agg_metrics(observation_df=train_df_obs,                                                                                customer_risk_at_product_lvl=train_customer_risk_at_product_lvl,
                                                                                use_clipped=use_clipped)
            test_final_customer_lvl_features = self.create_customer_lvl_agg_metrics(observation_df=test_df_obs,                                                                                    customer_risk_at_product_lvl=test_customer_risk_at_product_lvl,
                                                                                use_clipped=use_clipped)

            # 5. Define the churn:
            train_final_customer_lvl_features = self.define_churn_label(df=train_df,                                                                           final_customer_lvl_features=train_final_customer_lvl_features)
            test_final_customer_lvl_features = self.define_churn_label(df=test_df,                                                                          final_customer_lvl_features=test_final_customer_lvl_features)

        
            # Making the directory for the clipped data:
            os.makedirs(os.path.dirname(self.feature_engineering_config.clipped_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.feature_engineering_config.unclipped_train_file_path), exist_ok=True)
            if use_clipped:
                clipped_train_file_path = self.feature_engineering_config.clipped_train_file_path
                clipped_test_file_path = self.feature_engineering_config.clipped_test_file_path
                train_final_customer_lvl_features.to_parquet(clipped_train_file_path)
                test_final_customer_lvl_features.to_parquet(clipped_test_file_path)
                logging.info(f"Clipped training data saved to: {clipped_train_file_path}")
                logging.info(f"Clipped testing data saved to: {clipped_test_file_path}")
            else:
                unclipped_train_file_path = self.feature_engineering_config.unclipped_train_file_path
                unclipped_test_file_path = self.feature_engineering_config.unclipped_test_file_path
                train_final_customer_lvl_features.to_parquet(unclipped_train_file_path)
                test_final_customer_lvl_features.to_parquet(unclipped_test_file_path)
                logging.info(f"Unclipped training data saved to: {unclipped_train_file_path}")
                logging.info(f"Unclipped testing data saved to: {unclipped_test_file_path}")

            # Generate the artifacts:
            if use_clipped:
                feature_engineering_artifacts = FeatureEngineeringArtifacts(
                clipped_training_file_path = clipped_train_file_path,
                clipped_testing_file_path = clipped_test_file_path)
            else:
                feature_engineering_artifacts = FeatureEngineeringArtifacts(
                    unclipped_training_file_path = unclipped_train_file_path,
                    unclipped_testing_file_path = unclipped_test_file_path
                )
            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)
            logging.info(f"Feature Engineering Artifacts:\n{feature_engineering_artifacts}\n")
            logging.info(f"Feature Engineering Completed | Total Execution Time: {execution_time} min.")  
            return feature_engineering_artifacts
        except Exception as e:
            raise CustomerChurnException(e, sys)
    