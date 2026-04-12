import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from src.customer_churn.exception.exception import CustomerChurnException




class ShapVisualization:
    
    def __init__(self, results:dict):
        self.results = results

    
    def df_from_predict_api_response(self) -> pd.DataFrame:
        """
        Takes in the response of the predict endpoint, 
        converts it to a pandas dataframe with an extra column 'color' for visualization purpose
        """
        try:
            contributions = self.results['shap_values']
            sorted_contrib = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
            features = [c['feature'] for c in sorted_contrib]
            values = [c['shap_value'] for c in sorted_contrib]
            input_features = [c['value'] for c in sorted_contrib]
            shap_df= pd.DataFrame({'Features':features, 'SHAP_values': values, 'Input_features':input_features})
            shap_df['Color'] = ['Decreases churn risk ↓' if v < 0 else 'Increases churn risk ↑' for v in shap_df['SHAP_values']]
            shap_df = shap_df.sort_values('SHAP_values', key=abs, ascending=False)
            return shap_df
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def plot_shap_feature_importance(self):
        """
        Takes in the SHAP df and visualizes the bar graph based on the SHAP values to show the differences,
        where negative(-) SHAP value decreases the churn risk and positive(+) SHAP value increases the churn risk:
        Parameters:
        ----------
        df: a pandas dataframe consisting of the feature names, SHAP values, input features, and color column for the bar chart
        """
        try:
            shap_df = self.df_from_predict_api_response()
            net_shap = sum(shap_val for shap_val in shap_df['SHAP_values'])
            fig = px.bar(shap_df, 
                     x='Features', 
                     y='SHAP_values', 
                     custom_data= ['Input_features', 'Color'],
                     color='Color', 
                     width=1450,
                     height=600,
                     color_discrete_map={
                        'Increases churn risk ↑': '#b11346',
                        'Decreases churn risk ↓': '#0e7337'
                },
                title='SHAP Feature Importance',
                labels={'Color': 'Effect on Churn Risk',
                       'SHAP_values': 'SHAP Value (Impact on Churn Probability)',
                       'Features': 'Customer Feature'})
            fig.update_traces(
                hovertemplate = "<b>Impact:</b> %{customdata[1]}<br>" +
                                "<b>Feature:</b> %{x}<br>"+
                                "<b>Feature value:</b> %{customdata[0]}<br>"+
                                "<b>SHAP value:</b> %{y}<br>"+
                                "<extra></extra>"
            )
            return fig, net_shap, shap_df
        except Exception as e:
            raise CustomerChurnException(e, sys)
    


