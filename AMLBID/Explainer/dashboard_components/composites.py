__all__ = [
    'ImportancesComposite',
    'ClassifierModelStatsComposite',
    'RegressionModelStatsComposite',
    'IndividualPredictionsComposite',
    'ShapDependenceComposite',
    'ShapInteractionsComposite',
    'DecisionTreesComposite',
    'WhatIfComposite',
    'Testcomposite',
    'SuggestedModelComposite',
    'RefinementComposite',
    'DataProfiling',
    'DataProfilingmed',
    'ExportDash'
]
import os.path
import pandas as pd
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash
from ..AMLBID_Explainer import RandomForestExplainer, XGBExplainer
from ..dashboard_methods import *
from .classifier_components import *
from .regression_components import *
from .overview_components import *
from .connectors import *
from .shap_components import *
from .decisiontree_components import *
from dash.dependencies import Input, Output, State
from .ConfGenerator import *
from .. import to_html



    
class ImportancesComposite(ExplainerComponent):
    def __init__(self, explainer, title="Features Dependence", name=None,
                    hide_importances=False,hide_title=False,
                    hide_selector=True,depth=None, **kwargs):
        """Overview tab of feature importances

        Can show both permutation importances and mean absolute shap values.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Importances".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_importances (bool, optional): hide the ImportancesComponent
            hide_selector (bool, optional): hide the post label selector. 
                Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.importances = ImportancesComponent(
                explainer, name=self.name+"0", hide_selector=hide_selector, **kwargs)
        
        self.shap_summary = ShapSummaryComponent(
                    explainer, name=self.name+"0",
                    **update_params(kwargs, hide_selector=hide_selector, depth=5, hide_cats=True,hide_title=True))
        self.shap_dependence = ShapDependenceComponent(
                    explainer, name=self.name+"1",
                    hide_selector=hide_selector,**update_params(kwargs,  hide_cats=True,hide_title=False))
        self.connector = ShapSummaryDependenceConnector(
                    self.shap_summary, self.shap_dependence)
        

        self.register_components()

    def layout(self):
        return html.Div([            
                dbc.Row(dbc.Col([
                dbc.CardDeck([
                dbc.Card([
                dbc.CardHeader([
                    # html.H4([dbc.Button("Description", id="positioned-toast-toggle", color="primary", className="mr-1")],style={"float": "right"}),
                    html.H3(["Feature Importances"], className="card-title"),
                    html.H6("Which features had the biggest impact?",className="card-subtitle")]),
                dbc.CardBody([
                
                 dbc.Toast(html.Div([html.P(
            "On the plot, you can check out for yourself which parameters were the most important."
            f"{self.explainer.columns_ranked_by_shap(cats=True)[0]} was the most important"
            f", followed by {self.explainer.columns_ranked_by_shap(cats=True)[1]}"
            f" and {self.explainer.columns_ranked_by_shap(cats=True)[2]}."),
            #html.Br(),
            html.P("If you select 'detailed' summary type you can see the impact of that variable on "
            "each individual prediction. With 'aggregate' you see the average impact size "
            "of that variable on the finale prediction.")],style={"text-align": "justify"}),
            id="positioned-toast",header="Feature Importances",is_open=False,dismissable=True,
            style={"position": "fixed", "top": 25, "right": 10, "width": 400},),
            
                self.shap_summary.layout(),
                 
   
                ],style=dict(marginTop= -20))
           ]) , self.shap_dependence.layout(), ])

      ])) ], style=dict(marginTop=25, marginBottom=25) )

    # def component_callbacks(self, app):
    #     @app.callback(Output("positioned-toast", "is_open"),[Input("positioned-toast-toggle", "n_clicks")],)
    #     def open_toast(n):
    #         if n:
    #             return True
    #         return False
    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [to_html.hide(self.shap_summary.to_html(state_dict, add_header=False), self.hide_importances)]
        )
        if add_header:
            return to_html.add_header(html)
        return html

class ClassifierModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Obtained Performances", name=None,
                    hide_title=True, hide_selector=True, 
                    hide_globalcutoff=True,
                    hide_modelsummary=False, hide_confusionmatrix=False,
                    hide_precision=True, hide_classification=False,
                    hide_rocauc=False, hide_prauc=True,
                    hide_liftcurve=True, hide_cumprecision=True,hide_range=False,

                    
                    pos_label=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5, **kwargs):
        """Composite of multiple classifier related components: 
            - precision graph
            - confusion matrix
            - lift curve
            - classification graph
            - roc auc graph
            - pr auc graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.          
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_globalcutoff (bool, optional): hide CutoffPercentileComponent
            hide_modelsummary (bool, optional): hide ClassifierModelSummaryComponent
            hide_confusionmatrix (bool, optional): hide ConfusionMatrixComponent
            hide_precision (bool, optional): hide PrecisionComponent
            hide_classification (bool, optional): hide ClassificationComponent
            hide_rocauc (bool, optional): hide RocAucComponent
            hide_prauc (bool, optional): hide PrAucComponent
            hide_liftcurve (bool, optional): hide LiftCurveComponent
            hide_cumprecision (bool, optional): hide CumulativePrecisionComponent
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            bin_size (float, optional): bin_size for precision plot. Defaults to 0.1.
            quantiles (int, optional): number of quantiles for precision plot. Defaults to 10.
            cutoff (float, optional): initial cutoff. Defaults to 0.5.
        """
        super().__init__(explainer, title, name)

        self.summary = ClassifierModelSummaryComponent(explainer, name=self.name+"0", 
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.precision = PrecisionComponent(explainer, name=self.name+"1",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.confusionmatrix = ConfusionMatrixComponent(explainer, name=self.name+"2",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.cumulative_precision = CumulativePrecisionComponent(explainer, name=self.name+"3",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.liftcurve = LiftCurveComponent(explainer, name=self.name+"4",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.classification = ClassificationComponent(explainer, name=self.name+"5",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.rocauc = RocAucComponent(explainer, name=self.name+"6",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.prauc = PrAucComponent(explainer, name=self.name+"7",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)

        self.cutoffpercentile = CutoffPercentileComponent(explainer, name=self.name+"8",
                hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.cutoffconnector = CutoffConnector(self.cutoffpercentile,
                [self.summary, self.precision, self.confusionmatrix, self.liftcurve, 
                 self.cumulative_precision, self.classification, self.rocauc, self.prauc])

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                     html.H2('Model Performance:')]), hide=self.hide_title),
            ]),
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        self.cutoffpercentile.layout(),
                    ]), hide=self.hide_globalcutoff),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.summary.layout(), hide=self.hide_modelsummary),
                make_hideable(self.confusionmatrix.layout(), hide=self.hide_confusionmatrix),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.rocauc.layout(), hide=self.hide_rocauc),
                make_hideable(self.classification.layout(), hide=self.hide_classification)
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.precision.layout(), hide=self.hide_precision),
                make_hideable(self.prauc.layout(), hide=self.hide_prauc),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                make_hideable(self.liftcurve.layout(), self.hide_liftcurve),
                make_hideable(self.cumulative_precision.layout(), self.hide_cumprecision),
            ], style=dict(marginBottom=25)),
        ])
    
    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [to_html.hide(self.summary.to_html(state_dict, add_header=False), hide=self.hide_modelsummary),
             to_html.hide(self.confusionmatrix.to_html(state_dict, add_header=False), hide=self.hide_confusionmatrix)],
            [to_html.hide(self.precision.to_html(state_dict, add_header=False), hide=self.hide_precision), 
             to_html.hide(self.prauc.to_html(state_dict, add_header=False), hide=self.hide_prauc)],
            [to_html.hide(self.rocauc.to_html(state_dict, add_header=False), hide=self.hide_rocauc),
             to_html.hide(self.classification.to_html(state_dict, add_header=False), hide=self.hide_classification)],
            [to_html.hide(self.liftcurve.to_html(state_dict, add_header=False), hide=self.hide_liftcurve),
             to_html.hide(self.cumulative_precision.to_html(state_dict, add_header=False), hide=self.hide_cumprecision)]
        )
        if add_header:
            return to_html.add_header(html)
        return html


class RegressionModelStatsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Regression Stats", name=None,
                    hide_title=True, hide_modelsummary=False,
                    hide_predsvsactual=False, hide_residuals=False, 
                    hide_regvscol=False,
                    logs=False, pred_or_actual="vs_pred", residuals='difference',
                    col=None, **kwargs):
        """Composite for displaying multiple regression related graphs:

        - predictions vs actual plot
        - residual plot
        - residuals vs feature

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Regression Stats".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.
            hide_modelsummary (bool, optional): hide RegressionModelSummaryComponent
            hide_predsvsactual (bool, optional): hide PredictedVsActualComponent
            hide_residuals (bool, optional): hide ResidualsComponent
            hide_regvscol (bool, optional): hide RegressionVsColComponent
            logs (bool, optional): Use log axis. Defaults to False.
            pred_or_actual (str, optional): plot residuals vs predictions 
                        or vs y (actual). Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional): 
                    How to calcualte residuals. Defaults to 'difference'.
            col ({str, int}, optional): Feature to use for residuals plot. Defaults to None.
        """
        super().__init__(explainer, title, name)
     
        assert pred_or_actual in ['vs_actual', 'vs_pred'], \
            "pred_or_actual should be 'vs_actual' or 'vs_pred'!"

        self.modelsummary = RegressionModelSummaryComponent(explainer, 
                                name=self.name+"0",**kwargs)
        self.preds_vs_actual = PredictedVsActualComponent(explainer, name=self.name+"0",
                    logs=logs, **kwargs)
        self.residuals = ResidualsComponent(explainer, name=self.name+"1",
                    pred_or_actual=pred_or_actual, residuals=residuals, **kwargs)
        self.reg_vs_col = RegressionVsColComponent(explainer, name=self.name+"2",
                    logs=logs, **kwargs)

    def layout(self):
        return html.Div([
            dbc.Row([
                make_hideable(
                    dbc.Col([
                        html.H2('Model Performance:')]), hide=self.hide_title)
            ]),
            dbc.CardDeck([
                make_hideable(self.modelsummary.layout(), hide=self.hide_modelsummary),
                make_hideable(self.preds_vs_actual.layout(), hide=self.hide_predsvsactual),
            ], style=dict(margin=25)),
            dbc.CardDeck([
                make_hideable(self.residuals.layout(), hide=self.hide_residuals),
                make_hideable(self.reg_vs_col.layout(), hide=self.hide_regvscol),
            ], style=dict(margin=25))
        ])
    
    def to_html(self, state_dict=None, add_header=True):
        html = to_html.hide(to_html.title(self.title), hide=self.hide_title)
        html += to_html.card_rows(
            [to_html.hide(self.modelsummary.to_html(state_dict, add_header=False), hide=self.hide_modelsummary),
             to_html.hide(self.preds_vs_actual.to_html(state_dict, add_header=False), hide=self.hide_predsvsactual)],
            [to_html.hide(self.residuals.to_html(state_dict, add_header=False), hide=self.hide_residuals),
             to_html.hide(self.reg_vs_col.to_html(state_dict, add_header=False), hide=self.hide_regvscol)],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class IndividualPredictionsComposite(ExplainerComponent):
    def __init__(self, explainer, title="Individual Predictions", name=None,
                        hide_predindexselector=False, hide_predictionsummary=False,
                        hide_contributiongraph=False, hide_pdp=False,
                        hide_contributiontable=False,
                        hide_title=False, hide_selector=True, **kwargs):
        """Composite for a number of component that deal with individual predictions:

        - random index selector
        - prediction summary
        - shap contributions graph
        - shap contribution table
        - pdp graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Individual Predictions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_predindexselector (bool, optional): hide ClassifierRandomIndexComponent 
                or RegressionRandomIndexComponent
            hide_predictionsummary (bool, optional): hide ClassifierPredictionSummaryComponent
                or RegressionPredictionSummaryComponent
            hide_contributiongraph (bool, optional): hide ShapContributionsGraphComponent
            hide_pdp (bool, optional): hide PdpComponent
            hide_contributiontable (bool, optional): hide ShapContributionsTableComponent
            hide_title (bool, optional): hide title. Defaults to False.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)

        if self.explainer.is_classifier:
            self.index = ClassifierRandomIndexComponent(explainer, name=self.name+"0",
                    hide_selector=hide_selector, **kwargs)
            self.summary = ClassifierPredictionSummaryComponent(explainer, name=self.name+"1",
                    hide_selector=hide_selector, **kwargs)
        elif self.explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, name=self.name+"0",
                    hide_selector=hide_selector, **kwargs)
            self.summary = RegressionPredictionSummaryComponent(explainer, name=self.name+"1",
                    hide_selector=hide_selector, **kwargs)

        self.contributions = ShapContributionsGraphComponent(explainer, name=self.name+"2",
                        hide_selector=hide_selector, **kwargs)
        self.pdp = PdpComponent(explainer, name=self.name+"3",
                        hide_selector=hide_selector, **kwargs)
        self.contributions_list = ShapContributionsTableComponent(explainer, name=self.name+"4",
                        hide_selector=hide_selector,  **kwargs)

        self.index_connector = IndexConnector(self.index, 
                [self.summary, self.contributions, self.pdp, self.contributions_list])

    def layout(self):
        return dbc.Container([
                dbc.CardDeck([
                    make_hideable(self.index.layout(), hide=self.hide_predindexselector),
                    make_hideable(self.summary.layout(), hide=self.hide_predictionsummary),
                ], style=dict(marginBottom=25, marginTop=25)),
                dbc.CardDeck([
                    make_hideable(self.contributions.layout(), hide=self.hide_contributiongraph),
                    make_hideable(self.pdp.layout(), hide=self.hide_pdp),
                ], style=dict(marginBottom=25, marginTop=25)),
                dbc.Row([
                    dbc.Col([
                        make_hideable(self.contributions_list.layout(), hide=self.hide_contributiontable),
                    ], md=6),
                    dbc.Col([
                        html.Div([]),
                    ], md=6),
                ])
        ], fluid=True)
    
    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [to_html.hide(self.index.to_html(state_dict, add_header=False), self.hide_predindexselector), 
             to_html.hide(self.summary.to_html(state_dict, add_header=False), self.hide_predictionsummary)],
            [to_html.hide(self.contributions.to_html(state_dict, add_header=False), self.hide_contributiongraph),
             to_html.hide(self.pdp.to_html(state_dict, add_header=False), self.hide_pdp)],
            [to_html.hide(self.contributions_list.to_html(state_dict, add_header=False), self.hide_contributiontable)]
        )
        if add_header:
            return to_html.add_header(html)
        return html



class WhatIfComposite(ExplainerComponent):
    def __init__(self, explainer, title="What-If Analysis", name=None,
                        hide_whatifindexselector=False, hide_inputeditor=False,
                        hide_whatifprediction=False, hide_whatifcontributiongraph=False, 
                        hide_whatifpdp=True, hide_whatifcontributiontable=False,
                        hide_title=True, hide_selector=True, 
                        n_input_cols=4, sort='importance', **kwargs):
        """Composite for the whatif component:

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Individual Predictions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.
            hide_selector(bool, optional): hide all pos label selectors. Defaults to True.
            hide_whatifindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_inputeditor (bool, optional): hide FeatureInputComponent
            hide_whatifprediction (bool, optional): hide PredictionSummaryComponent
            hide_whatifcontributiongraph (bool, optional): hide ShapContributionsGraphComponent
            hide_whatifcontributiontable (bool, optional): hide ShapContributionsTableComponent
            hide_whatifpdp (bool, optional): hide PdpComponent
            n_input_cols (int, optional): number of columns to divide the feature inputs into.
                Defaults to 4. 
            sort ({'abs', 'high-to-low', 'low-to-high', 'importance'}, optional): sorting of shap values. 
                        Defaults to 'importance'.
        """
        super().__init__(explainer, title, name)
        
        if 'hide_whatifcontribution' in kwargs:
            print("Warning: hide_whatifcontribution will be deprecated, use hide_whatifcontributiongraph instead!")
            self.hide_whatifcontributiongraph = kwargs['hide_whatifcontribution']

        self.input = FeatureInputComponent(explainer, name=self.name+"0",
                        hide_selector=hide_selector,hide_title=True, hide_subtitle=True, n_input_cols=self.n_input_cols,
                        **update_params(kwargs, hide_index=True))
        
        if self.explainer.is_classifier:
                        
            self.index = ClassifierRandomIndexComponentPerso(explainer, name=self.name+"1",
                    hide_selector=hide_selector,hide_title=True, hide_subtitle=True, 
                    hide_slider=True,hide_pred_or_perc=True,hide_labels=True, **kwargs)
                    
            self.prediction = ClassifierPredictionSummaryComponent(explainer, name=self.name+"2",
                        feature_input_component=self.input,
                        hide_star_explanation=True,
                        hide_selector=hide_selector, **kwargs)
        elif self.explainer.is_regression:
            pass
            #self.index = RegressionRandomIndexComponent(explainer, name=self.name+"1", **kwargs)
            #self.prediction = RegressionPredictionSummaryComponent(explainer, name=self.name+"2",
              #           feature_input_component=self.input, **kwargs)
        
        
        self.contribgraph = ShapContributionsGraphComponent(explainer, name=self.name+"3",
                        feature_input_component=self.input,
                        hide_selector=hide_selector, sort=sort, **kwargs)
        self.contribtable = ShapContributionsTableComponentPerso(explainer, name=self.name+"4",
                        feature_input_component=self.input,hide_cats=True,
                        hide_selector=hide_selector, sort=sort, **kwargs)
        
        self.pdp = PdpComponent(explainer, name=self.name+"5",
                        feature_input_component=self.input,
                        hide_selector=hide_selector, **kwargs)

        self.index_connector = IndexConnector(self.index, [self.input])

    def layout(self):
        return dbc.Container([
                dbc.Row([
                    make_hideable(
                        dbc.Col([html.H1(self.title)]), hide=self.hide_title),
                        ]),

                dbc.Row([
                    make_hideable(
                        dbc.Col([
                           dbc.Card([
                               dbc.CardHeader([html.H4("Select Observation", className="card-title"),
                                               html.H6("Select from list or pick at random", className="card-subtitle")]),
                               dbc.CardBody([
                                   self.index.layout(),
                                   html.Hr(),
                                   self.input.layout()
                                           ],style=dict(marginTop= -20))])
                                 ], md=7), hide=self.hide_whatifindexselector),
                    make_hideable(
                        dbc.Col([
                            self.prediction.layout(),
                        ], md=5), hide=self.hide_whatifprediction),
                ], style=dict(marginBottom=15, marginTop=15)),
                dbc.CardDeck([
                    #make_hideable(self.contribgraph.layout(), hide=self.hide_whatifcontributiongraph),
                    make_hideable(self.pdp.layout(), hide=self.hide_whatifpdp),
                ], style=dict(marginBottom=15, marginTop=15)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.contribtable.layout()
                        ], md=6), hide=self.hide_whatifcontributiontable),
                    dbc.Col([self.contribgraph.layout()], style=dict(marginBottom=15), md=6),
                ])
        ], fluid=True)
    
    def to_html_(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [to_html.hide(self.index.to_html(state_dict, add_header=False), self.hide_whatifindexselector), 
             to_html.hide(self.prediction.to_html(state_dict, add_header=False), self.hide_whatifprediction)],
            [to_html.hide(self.input.to_html(state_dict, add_header=False), self.hide_inputeditor)],
            [to_html.hide(self.contribgraph.to_html(state_dict, add_header=False), self.hide_whatifcontributiongraph),
             to_html.hide(self.pdp.to_html(state_dict, add_header=False), self.hide_whatifpdp)],
            [to_html.hide(self.contribtable.to_html(state_dict, add_header=False), self.hide_whatifcontributiontable)]
        )
        html = to_html.div(html)
        if add_header:
            return to_html.add_header(html)
        return html


class ShapDependenceComposite(ExplainerComponent):
    def __init__(self, explainer, title='Feature Dependence', name=None,
                    hide_selector=True, 
                    hide_shapsummary=False, hide_shapdependence=False,
                    depth=None, cats=True, **kwargs):
        """Composite of ShapSummary and ShapDependence component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Dependence".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_shapsummary (bool, optional): hide ShapSummaryComponent
            hide_shapdependence (bool, optional): ShapDependenceComponent
            depth (int, optional): Number of features to display. Defaults to None.
            cats (bool, optional): Group categorical features. Defaults to True.
        """
        super().__init__(explainer, title, name)
        
        self.shap_summary = ShapSummaryComponent(
                    self.explainer, name=self.name+"0",
                    **update_params(kwargs, hide_selector=hide_selector, depth=depth, cats=cats))
        self.shap_dependence = ShapDependenceComponent(
                    self.explainer, name=self.name+"1",
                    hide_selector=hide_selector, cats=cats,
                    **update_params(kwargs, hide_cats=True)
                    )
        self.connector = ShapSummaryDependenceConnector(
                    self.shap_summary, self.shap_dependence)

    def layout(self):
        return dbc.Container([
            dbc.CardDeck([
                make_hideable(self.shap_summary.layout(), hide=self.hide_shapsummary),
                make_hideable(self.shap_dependence.layout(), hide=self.hide_shapdependence),
            ], style=dict(marginTop=25)),
        ], fluid=True)
    
    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [to_html.hide(self.shap_summary.to_html(state_dict, add_header=False), self.hide_shapsummary), 
             to_html.hide(self.shap_dependence.to_html(state_dict, add_header=False), self.hide_shapdependence)],
        )
        if add_header:
            return to_html.add_header(html)
        return html


class ShapInteractionsComposite(ExplainerComponent):
    def __init__(self, explainer, title='Feature Interactions', name=None,
                    hide_selector=True,
                    hide_interactionsummary=False, hide_interactiondependence=False,
                    depth=None, cats=True, **kwargs):
        """Composite of InteractionSummaryComponent and InteractionDependenceComponent

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Feature Interactions".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_interactionsummary (bool, optional): hide InteractionSummaryComponent
            hide_interactiondependence (bool, optional): hide InteractionDependenceComponent
            depth (int, optional): Initial number of features to display. Defaults to None.
            cats (bool, optional): Initally group cats. Defaults to True.
        """
        super().__init__(explainer, title, name)

        self.interaction_summary = InteractionSummaryComponent(explainer, name=self.name+"0",
                hide_selector=hide_selector, depth=depth, cats=cats, **kwargs)
        self.interaction_dependence = InteractionDependenceComponent(explainer, name=self.name+"1",
                hide_selector=hide_selector, cats=cats, **update_params(kwargs, hide_cats=True))
        self.connector = InteractionSummaryDependenceConnector(
            self.interaction_summary, self.interaction_dependence)
        
    def layout(self):
        return dbc.Container([
                dbc.CardDeck([
                    make_hideable(self.interaction_summary.layout(), hide=self.hide_interactionsummary),
                    make_hideable(self.interaction_dependence.layout(), hide=self.hide_interactiondependence),
                ], style=dict(marginTop=25))
        ], fluid=True)

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [to_html.hide(self.interaction_summary.to_html(state_dict, add_header=False), self.hide_interactionsummary), 
             to_html.hide(self.interaction_dependence.to_html(state_dict, add_header=False), self.hide_interactiondependence)],
        )
        if add_header:
            return to_html.add_header(html)
        return html

class DecisionTreesComposite(ExplainerComponent):
    def __init__(self, explainer, title="Decision Path", name=None,
                    hide_treeindexselector=False, hide_treesgraph=True,
                    hide_treepathtable=True, hide_treepathgraph=False,
                    hide_selector=True,n_input_cols=4, sort='importance', **kwargs):
        """Composite of decision tree related components:
        
        - index selector
        - individual decision trees barchart
        - decision path table
        - deciion path graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        RandomForestClassifierExplainer() or RandomForestRegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_treeindexselector (bool, optional): hide ClassifierRandomIndexComponent
                or RegressionRandomIndexComponent
            hide_treesgraph (bool, optional): hide DecisionTreesComponent
            hide_treepathtable (bool, optional): hide DecisionPathTableComponent
            hide_treepathgraph (bool, optional): DecisionPathGraphComponent
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
        """
        super().__init__(explainer, title, name)
        #self.input = FeatureInputComponent(explainer, name=self.name+"4",
                        #hide_selector=hide_selector, n_input_cols=self.n_input_cols,hide_title=True,
                        #**update_params(kwargs, hide_index=True))
                        
        self.trees = DecisionTreesComponent(explainer, name=self.name+"0",
                    hide_selector=hide_selector, **kwargs)
        self.decisionpath_table = DecisionPathTableComponent(explainer, name=self.name+"1",
                    hide_selector=hide_selector, **kwargs)

        if explainer.is_classifier:
            self.index = ClassifierRandomIndexComponentPerso(explainer, name=self.name+"2",
                    hide_selector=hide_selector,hide_title=True, hide_subtitle=True, 
                    hide_slider=True,hide_pred_or_perc=True,hide_labels=True, **kwargs)
        elif explainer.is_regression:
            self.index = RegressionRandomIndexComponent(explainer, name=self.name+"2",
                    **kwargs)

        self.prediction = ClassifierPredictionSummaryComponentPerso(explainer, name=self.name+"4",
                        hide_star_explanation=True,hide_title=True,
                        hide_selector=True, **kwargs)                
        self.decisionpath_graph = DecisionPathGraphComponent(explainer, name=self.name+"3",
                    hide_selector=hide_selector, **kwargs)
        
        self.contribtable = ShapContributionsTableComponentPerso(explainer, name=self.name+"5",
                        hide_index=True,hide_cats=True, depth=2,
                        hide_selector=hide_selector, sort=sort, **kwargs)

        self.index_connector = IndexConnector(self.index, 
            [self.trees, self.decisionpath_table, self.decisionpath_graph,self.prediction,self.contribtable] )
        
        self.highlight_connector = HighlightConnector(self.trees, 
            [self.decisionpath_table, self.decisionpath_graph])

    def layout(self):
        if isinstance(self.explainer, XGBExplainer):
            return html.Div([
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.index.layout()
                        ]), hide=False), 

                        
                ], style=dict(margin=25)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.trees.layout(), 
                        ], md=8), hide=self.hide_treesgraph),
                    make_hideable(
                        dbc.Col([
                            self.decisionpath_table.layout(), 
                        ], md=4), hide=True),
                ], style=dict(margin=25)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.decisionpath_graph.layout()
                        ]), hide=self.hide_treepathgraph),
                ], style=dict(margin=25)),
            ])
        elif isinstance(self.explainer, RandomForestExplainer):
            return html.Div([
                dbc.Row([
                        dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([html.H4("Select Observation", className="card-title"),
                                            html.H6("Select from list or pick at random", className="card-subtitle")]),
                            dbc.CardBody([
                            self.index.layout(),
                            self.prediction.layout(),
                                        ],style=dict(marginTop= -20))])
                        ], md=15),
                    dbc.Col([
                            self.decisionpath_graph.layout()
                        ]),
                        
                                            
                ], style=dict(margin=25, marginBottom=0)),
                
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.trees.layout(), 
                        ]), hide=self.hide_treesgraph),
                ], style=dict(margin=0)),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            self.decisionpath_table.layout(), 
                        ]), hide=True),
                ], style=dict(margin=0)),
                dbc.Row([
                    make_hideable(
                         dbc.Col([self.contribtable.layout()]),
                         hide=True),
                ], style=dict(marginBottom=25,marginTop=25)),
            ])
        else:
            raise ValueError("explainer is neither a RandomForestExplainer nor an XGBExplainer! "
                            "Pass decision_trees=False to disable the decision tree tab.")

    def to_html_(self, state_dict=None, add_header=True):
        html = to_html.title(self.title)
        html += to_html.card_rows(
            [to_html.hide(self.index.to_html(state_dict, add_header=False), self.hide_treeindexselector)],
            [to_html.hide(self.trees.to_html(state_dict, add_header=False), self.hide_treesgraph)],
            [to_html.hide(self.decisionpath_table.to_html(state_dict, add_header=False), self.hide_treepathtable)],
        )
        if add_header:
            return to_html.add_header(html)
        return html
        
        
class SuggestedModelComposite(ExplainerComponent):
    def __init__(self, explainer, title="Suggested Model", name=None,
                    hide_title=True, hide_selector=True, 
                    hide_globalcutoff=False,
                    hide_modelsummary=False, hide_confusionmatrix=False,
                    hide_precision=False, hide_classification=False,
                    hide_rocauc=False, hide_prauc=False,
                    hide_liftcurve=False, hide_cumprecision=False,
                    pos_label=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5, **kwargs):
        """Composite of multiple classifier related components: 
            - precision graph
            - confusion matrix
            - lift curve
            - classification graph
            - roc auc graph
            - pr auc graph

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to 
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements. 
                        If None then random uuid is generated to make sure 
                        it's unique. Defaults to None.
            hide_title (bool, optional): hide title. Defaults to True.          
            hide_selector (bool, optional): hide all pos label selectors. Defaults to True.
            hide_globalcutoff (bool, optional): hide CutoffPercentileComponent
            hide_modelsummary (bool, optional): hide ClassifierModelSummaryComponent
            hide_confusionmatrix (bool, optional): hide ConfusionMatrixComponent
            hide_precision (bool, optional): hide PrecisionComponent
            hide_classification (bool, optional): hide ClassificationComponent
            hide_rocauc (bool, optional): hide RocAucComponent
            hide_prauc (bool, optional): hide PrAucComponent
            hide_liftcurve (bool, optional): hide LiftCurveComponent
            hide_cumprecision (bool, optional): hide CumulativePrecisionComponent
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            bin_size (float, optional): bin_size for precision plot. Defaults to 0.1.
            quantiles (int, optional): number of quantiles for precision plot. Defaults to 10.
            cutoff (float, optional): initial cutoff. Defaults to 0.5.
        """
        super().__init__(explainer, title, name)

        
    
    


    def layout(self):
        
        ModelDescription=pd.read_pickle(os.path.dirname(__file__) +"/../assets/ModelsDescription.pkl")
        #MD=ModelDescription[(ModelDescription.id== "SVM")]
        MD=ModelDescription[(ModelDescription.Cname== self.explainer.model.__class__.__name__)]
        RecommendedConf=self.explainer.recommended_config[0][1]
        rows=[]
        table_header = [html.Thead(html.Tr([html.Th("Hyperparameter"), html.Th("Value")]))]
        for key,val in RecommendedConf.items():
            rows.append(html.Tr([html.Td(key), html.Td(str(val))]))        
        table_body = [html.Tbody(rows)]

        classifier_name=MD.Cname
        return html.Div([

            
           dbc.Row([
                dbc.Col([                  
                    dbc.Card([
                        dbc.CardHeader([html.H3(MD.Name, className="card-title")]),
                        dbc.CardBody([
                             html.Div([html.H6(MD.Cimport , style={"float": "left"}),html.Code(html.H6(html.B(MD.Cname))),
                             html.I(MD.Conceptual_desc.to_list()[0]),]),
                             
                             html.Br(),
                             html.P(MD.details),
                           
                                
                                    ],style={"text-align": "justify"}),
                        dbc.CardFooter([dbc.CardLink("Learn more>>", href=MD.Ref.to_list()[0], style={"float": "right"})]),
                            ]),
                        html.Br(),
                        html.Div(html.Img(src="./assets/AMLBID.png",style={"max-width":"60%", "height:":"auto"} ),style={ "margin-left": "200px"}),
                            
                        ], width=6),
                
                
                dbc.Col([ 
                    #html.Br(),
                    dbc.Card([
                        dbc.CardHeader([html.H3("Recommended model configuration", className="card-title")]),
                        dbc.CardBody([dbc.Table(table_header + table_body, bordered=False)]),
                        dbc.CardFooter([
                            html.Div([
                                 dbc.Button("Export Pipeline", id="example-button", color="info", className="mr-1", style={"float": "left"}),
                                 dbc.Tooltip(f"Export recommended configuration implementation as a Python file",target="example-button",placement="right", 
                                             style={"width":"300px"}),
                                 #html.Span(id="example-output", style={, style={"float": "right"}, style={"float": "right"}, style={"float": "right"}}),
                        dbc.Alert(["Configuration implementation exported ", html.B("successfully!")],color="success", id="alert-auto",is_open=False,duration=7000,
                                  style={"float": "right","margin-bottom":"0px"}),
                                 ]),
                                      ])
                             ]), html.Br(),
                        ]),
            ], style=dict(marginTop=25, marginBottom=25)   )
        ])

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        metrics_df = self.explainer.recommended_config[0][1]
        html = to_html.table_from_df(metrics_df)
        html = to_html.card(html, title=self.title+" : "+self.explainer.model.__class__.__name__)
        if add_header:
            return to_html.add_header(html)
        return html
    
    def component_callbacks(self, app):
        
                @app.callback(
                #Output("example-output", "children"), [Input("example-button", "n_clicks")],
                    
                Output("alert-auto", "is_open"),[Input("example-button", "n_clicks")],[State("alert-auto", "is_open")],
                )        
                def toggle_alert(n, is_open):
                    if n:
                        generate_pipeline_file(self.explainer.model.__class__.__name__,self.explainer.recommended_config,'your dataset path')
                        return not is_open
                    return is_open
        

class Testcomposite(ExplainerComponent):
    def __init__(self, explainer,title="Suggested configurations", name=None, **kwargs ):
        super().__init__(explainer, title,name)        
    
    def layout(self):
        DataComposite=self.explainer.recommended_config
        ModelDescription=pd.read_pickle(os.path.dirname(__file__) +"/../assets/ModelsDescription.pkl")
        def make_item(i,md,exp_acc,RecommendedConf,isHidden):
            rows=[]
            table_header = [html.Thead(html.Tr([html.Th("Hyperparameter"), html.Th("Value")]))]
            for key,val in RecommendedConf.items():
                rows.append(html.Tr([html.Td(key), html.Td(str(val))]))        
            table_body = [html.Tbody(rows)]

            return  make_hideable(dbc.Card([
                    html.Br(),
                    dbc.CardHeader([dbc.Form([dbc.FormGroup([
                       html.Tr([html.Th(dbc.Button(html.H5(f"Recommendation {i} : "+md.Cname),id=f"group-{i}-toggle",block=True,
                                                   style={"border": "none",  "background-color": "inherit",  "font-size": "16px",
                                                          "cursor": "pointer" , "color": "black", "width": "100%","align":"left",
                                                          "text-align":"left"}),style={"width":"600px"}),

                        html.Th(html.H5(f"Expected accuracy :    {exp_acc} ") ,style={"width":"400px"}),
                        html.Th([dbc.Button("Export Pipeline" ,id=f"example-button{i}",color="info"), 
                        dbc.Tooltip(f"Export recommended config as a Python file",target=f"example-button{i}",placement="top",
                                    style={"width":"300px"}),
                        dbc.Toast("Recommended configuration implementation exported successfully!",id=f"alert-auto{i}",
                                                    header="Export pipeline",is_open=False,dismissable=True,icon="success",duration=4000,
                                                    style={"position": "fixed", "top": 10, "right": 10, "width": 350}), 
                                         
                                         
                                       ], style={"width":"200px"})]),
                       ])  ,],inline=True)  ])  ,
                
                    dbc.Collapse([
                        
                        dbc.CardBody([
           dbc.Row([
                dbc.Col([                  
                    dbc.Card([
                        dbc.CardHeader([html.H3(md.Name, className="card-title")]),
                        dbc.CardBody([
                             html.Div([html.H6(md.Cimport , style={"float": "left"}),html.Code(html.H6(html.B(md.Cname))),
                             html.I(md.Conceptual_desc.to_list()[0]),]),
                             html.Br(),
                             html.P(md.details),
                           
                                
                                    ],style={"text-align": "justify"}),
                        dbc.CardFooter([dbc.CardLink("Learn more>>", href=md.Ref.to_list()[0], style={"float": "right"})]),
                            ]),
                        html.Br(),                            
                        ], width=6),
               
                    dbc.Col([ 
                    #html.Br(),
                    dbc.Card([
                        dbc.CardHeader([html.H3("Recommended model configuration", className="card-title")]),
                        dbc.CardBody([
                            dbc.Table(table_header + table_body, bordered=False)
                        
                        ]),
                            
                                      
                             ]), html.Br(),
                        ]),
            ],   )                           
                        ]) 
                                    
                                    
                    ],id=f"collapse-{i}"),
                ]),hide=isHidden)
        
        
        
        items=[html.Br()]
        if len(DataComposite)==3:
            for index, item in zip(range(len(DataComposite)), DataComposite):
                md=ModelDescription[(ModelDescription.Cname== item[0][1].__class__.__name__)]
                RecommendedConf=item[1]
                acc=round(item[2],5)
                items.append(make_item(index+1,md,acc,RecommendedConf,False))
                
        if len(DataComposite)==2:
            for index, item in zip(range(len(DataComposite)), DataComposite):
                md=ModelDescription[(ModelDescription.Cname== item[0][1].__class__.__name__)]
                RecommendedConf=item[1]
                acc=round(item[2],5)
                items.append(make_item(index+1,md,acc,RecommendedConf,False))
            items.append(make_item(index+2,md,acc,RecommendedConf,True))
            
        if len(DataComposite)==1:
            for index, item in zip(range(len(DataComposite)), DataComposite):
                md=ModelDescription[(ModelDescription.Cname== item[0][1].__class__.__name__)]
                RecommendedConf=item[1]
                acc=round(item[2],5)
                items.append(make_item(index+1,md,acc,RecommendedConf,False))
            items.append(make_item(index+2,md,acc,RecommendedConf,True))
            items.append(make_item(index+3,md,acc,RecommendedConf,True))
            
            
        return html.Div(items,
                        className="accordion", style={"margin-left":"100px","margin-right":"100px"})
    
#,html.Br(), make_item(2),html.Br(), make_item(3)

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        ModelDescription=pd.read_pickle(os.path.dirname(__file__) +"/../assets/ModelsDescription.pkl")
        DataComposite=self.explainer.recommended_config
        for index, item in zip(range(1), DataComposite):
                RecommendedConf=item[1]
                name=item[0][1].__class__.__name__
        RecommendedConf=item[1]
        metrics_df = RecommendedConf
        md=ModelDescription[(ModelDescription.Cname== self.explainer.model.__class__.__name__)]
        
        html0 = to_html.modelcard("title",md.Cimport.to_list()[0],md.Cname.to_list()[0],md.Conceptual_desc.to_list()[0],md.details.to_list()[0]  )
        html0 = to_html.card(html0, title=name)
        
        txt=pipeline_code(self.explainer.model.__class__.__name__,RecommendedConf,'your dataset path')
        # txt="from panadas import flkjh"
        htmlc = to_html.code(txt)
        htmlc = to_html.card(htmlc, title="Recommend pipeline code")
        
        
        ddd=pd.DataFrame({'Hyperparameter' : RecommendedConf.keys() , 'Value' : RecommendedConf.values() })
        html1 = to_html.table_from_df(ddd)
        html1 = to_html.card(html1, title="Recommended model configuration")
        
        html=to_html.card_rows([html0,html1],[htmlc])
        if add_header:
            return to_html.add_header(html)
        return html
    
    def component_callbacks(self, app):
        DataComposite=self.explainer.recommended_config   
        
        @app.callback(
            [Output(f"collapse-1", "is_open"),Output(f"collapse-2", "is_open"),Output(f"collapse-3", "is_open"),
             Output(f"alert-auto1", "is_open"),Output(f"alert-auto2", "is_open"),Output(f"alert-auto3", "is_open")],
            [Input(f"group-1-toggle", "n_clicks"),Input(f"group-2-toggle", "n_clicks"),Input(f"group-3-toggle", "n_clicks"),
             Input(f"example-button1", "n_clicks"),Input(f"example-button2", "n_clicks"),Input(f"example-button3", "n_clicks")],
            [State(f"collapse-1", "is_open"),State(f"collapse-2", "is_open"),State(f"collapse-3", "is_open"),
             State("alert-auto1", "is_open"),State("alert-auto2", "is_open"),State("alert-auto3", "is_open")],
        )
                
    
        def toggle_accordion(n1, n2, n3,n4,n5,n6, is_open1, is_open2, is_open3, is_open4, is_open5, is_open6):
            
            ctx = dash.callback_context

            if not ctx.triggered:
                return False, False, False,False, False, False
            else:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "group-1-toggle" and n1:
                return not is_open1, False, False,False, False, False
            elif button_id == "group-2-toggle" and n2:
                return False, not is_open2, False,False, False, False
            elif button_id == "group-3-toggle" and n3:
                return False, False, not is_open3,False, False, False
            elif button_id == "example-button1" and n4:
                item=DataComposite[0]
                generate_pipeline_file(item[0][1].__class__.__name__,item[1],'your dataset path')
                return  False, False, False,not is_open4,False, False
            elif button_id == "example-button2" and n5:
                item=DataComposite[1]
                generate_pipeline_file(item[0][1].__class__.__name__,item[1],'your dataset path')
                return False, False, False,False,not is_open5,False
            elif button_id == "example-button3" and n6:
                item=DataComposite[2]
                generate_pipeline_file(item[0][1].__class__.__name__,item[1],'your dataset path')
                return False, False,False, False, False, not is_open6
        return False, False, False,False, False, False
    

class RefinementComposite(ExplainerComponent):
    def __init__(self, explainer, title="Hyperparameters importance", name=None,
                    hide_title=True, hide_selector=True, 
                    hide_globalcutoff=False,
                    hide_modelsummary=False, hide_confusionmatrix=False,
                    hide_precision=False, hide_classification=False,
                    hide_rocauc=False, hide_prauc=False,
                    hide_liftcurve=False, hide_cumprecision=False,
                    pos_label=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5, **kwargs):

        super().__init__(explainer, title, name)


    def layout(self):
        dic={
                "AdaBoostClassifier": [4,7],
                "GradientBoostingClassifier": [7,11],
                "ExtraTreesClassifier": [5,12],
                "DecisionTreeClassifier": [4,7],
                "RandomForestClassifier": [5,12],
                "SVC": [6,11]
            }

        fAnova_data = pd.read_csv(os.path.dirname(__file__)+'/../assets/ANOVA_FINAL.csv',sep=',')
        NN=self.explainer.recommended_config[0][-1]
        CN=self.explainer.model.__class__.__name__
        RS=fAnova_data[(fAnova_data.dataset==NN) & (fAnova_data.algorithm =="RandomForest")].to_numpy()
        HI=pd.DataFrame(RS[:dic[CN][0]]).sort_values(by=[2], ascending=False).to_numpy()       
        
        
        
        hyper_importance_table_header = [html.Thead(html.Tr([html.Th("Hyperparameter"), html.Th("Importance")]))]

        rows=[]
        for i in range(dic[CN][0]):
            rows.append(html.Tr([html.Td(HI[i][1]), html.Td(dbc.Progress(value=HI[i][2]*100, color="00005E", className="mb-3"))]))

        hyper_importance_table_body = [html.Tbody(rows)]

        
        HCI=pd.DataFrame(RS[dic[CN][0]:2*dic[CN][0]]).sort_values(by=[2], ascending=False).to_numpy()
        rows=[]
        hyper_corr_importance_table_header = [html.Thead(html.Tr([html.Th("Hyperparameters"), html.Th("Dependence")]))]
        for i in range(dic[CN][0]):
            rows.append(html.Tr([html.Td(HCI[i][1]), html.Td(dbc.Progress(value=HCI[i][2]*2500, color="00005E", className="mb-3"))]))         
        hyper_corr_importance_table_body = [html.Tbody(rows)]
        
        
        
        
        return html.Div([

      dbc.Row([     
      dbc.Col([ html.Br(),
                    dbc.Card(
                        dbc.CardBody([
                
                    html.H3("Hyperparameters importance"),
                    dbc.Table(hyper_importance_table_header + hyper_importance_table_body, bordered=False),
                    #className="p-5"
                    #self.shap_dependence.layout()
                ])), html.Br(),
                ]),

            dbc.Col([ html.Br(),
                    dbc.Card(
                        dbc.CardBody([
                
                    html.H3("Hyperparameters correlation"),
                    dbc.Table(hyper_corr_importance_table_header + hyper_corr_importance_table_body, bordered=False),
                    #className="p-5"
                    #self.shap_dependence.layout()
                ])), html.Br(),
                ]),      
            
            
              ])   
            ]  )


    
    
    
#--------------------------------------------------------------------------------------------------------

class DataProfiling(ExplainerComponent):
    def __init__(self, explainer, title="Data profiling", name=None,
                    hide_title=True, hide_selector=True, 
                    hide_globalcutoff=False,
                    hide_modelsummary=False, hide_confusionmatrix=False,
                    hide_precision=False, hide_classification=False,
                    hide_rocauc=False, hide_prauc=False,
                    hide_liftcurve=False, hide_cumprecision=False,
                    pos_label=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5, **kwargs):

        super().__init__(explainer, title, name)


    def layout(self):
        
        return html.Div([
            
            dbc.Card([
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Overview", tab_id="Overview"),
                    dbc.Tab(label="Samples", tab_id="Samples"),
                    #dbc.Tab(label="Duplicate rows", tab_id="dup_rows"),
                ],
                id="card-tabs",
                card=True,
                active_tab="Overview",
            )
        ),
        dbc.CardBody(html.P(id="card-content", className="card-text")),
        ])
        ]  )
    
    



    def component_callbacks(self, app):
        def DataOverview (ds):
            metafeatures = {}
            #dataset general info
            n_var = ds.shape[1]
            n_obs = ds.shape[0]
            n_missing = ds.isnull().sum().sum() + ds.isna().sum().sum()
            n_classes = ds.iloc[:, -1].nunique()
            dup_rows =len(ds)-len(ds.drop_duplicates())
            #varibales (data type) infos
            Numeric = ds.select_dtypes(include='number').shape[1]
            Categorical = ds.select_dtypes(include='object').shape[1]
            Boolean = ds.select_dtypes(include='bool').shape[1]
            Date = ds.select_dtypes(include='datetime64').shape[1]
            Unsupported = 0            
            dsInfo=[n_var, n_obs, n_classes,n_missing,  dup_rows]
            varInfo=[Numeric, Categorical, Boolean, Date, Unsupported]
            return dsInfo,varInfo


        
        DataComposite=self.explainer.recommended_config    
        dsInfo,varInfo = DataOverview(self.explainer.Dataset)
        
        @app.callback(Output("card-content", "children"), [Input("card-tabs", "active_tab")])




        def tab_content(active_tab):
            value=1
            data = [
        {
            'values': [10,30,60],
            'type': 'pie',
        },
    ]
            
            
            ds_info = pd.DataFrame({
                    " ": ["Number of variables", "Number of observations","Number of classes", "Missing cells", "Duplicate rows"],
                   "  ": dsInfo,
                            })
            
            
            vr_type = pd.DataFrame({
                    " ": ["Numeric", "Categorical", "Boolean", "Date", "Unsupported"],
                   "  ": varInfo,
                
                            })
            
            
            dfgg = pd.DataFrame(
    {
        "First Name": ["Arthur", "Ford", "Momo", "Trillian","Moncef"],
        "Last Name": ["Dent", "Prefect", "Beeblebrox", "Astra",'GOURANI'],
    }
)

            
            if active_tab=="Overview":
                return html.Div([
                    dbc.Row([     

                         dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([dbc.Form([dbc.FormGroup([
                       html.H4("Dataset info", className="card-title"),
                       html.Div([dbc.Badge("Warning", color="warning",className="ml-1")], style={"float": "right"}),
                       ])  ],inline=True)  ])  ,
                                
                            dbc.CardBody([
                          dbc.Table.from_dataframe(ds_info, striped=True, bordered=True, hover=True),
                                dcc.Graph(
            id='graph',
            figure={ 'data': data ,"layout": {
                "title": "Classes distribution",
            "height": 400,  # px

        }, }) 
                                
                               
                                
                                
                                        ])])
                        ]),
                           
                        
             dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([html.H4("Data type", className="card-title"),
                                          ]),
                            dbc.CardBody([
                           dbc.Table.from_dataframe(vr_type, striped=True, bordered=True, hover=True),
                                        ])])
                        ]),
                        
                        
                        
              ]), 
                    
              
   
                       dbc.Row([
                       
                       dbc.Col([html.Div([
                html.H4("Warnings"),
                 dbc.Alert([
                "Dataset has ",
                html.A(dsInfo[4], href="#", className="alert-link"),
                html.A("("+str(round((dsInfo[4]*100)/dsInfo[1],2))+")%",  className="alert-link"),
                " duplicate rows"
                ],color="warning",style={"with": "500px"}),
                    
                dbc.Alert([
                "Dataset has ",
                html.A(dsInfo[3], href="#", className="alert-link"),
                html.A("("+str(round((dsInfo[3]*100)/(dsInfo[0]*dsInfo[1]),2))+")%",  className="alert-link"),
                " missing values"
                ],color="warning"),
                
                            dbc.Alert([
                "Your ",
                html.A('unbalanced dataset', href="#", className="alert-link"),
                " will bias the prediction model towards the more common class!"
                ],color="warning"),
                           
                           
                           
                           
             ],id="warnings")   ],md=6)]) ])
            
            
            if active_tab=="Samples":
                return dbc.Row([     
      dbc.Col([ html.Br(),
                    dbc.Card(
                        dbc.CardBody([
                
                    html.H3("Dataset samples extract"),
                            # dbc.Table.from_dataframe(self.explainer.Dataset.head(10), striped=True, bordered=True, hover=True),
                    #dbc.Table(hyper_importance_table_header + hyper_importance_table_body, bordered=False),
                    #className="p-5"
                    #self.shap_dependence.layout()
                ])), html.Br(),
                ]),])

                    
             
            if active_tab=="dup_rows":
                return dbc.Row([     
      dbc.Col([ html.Br(),
                    dbc.Card(
                        dbc.CardBody([
                
                    html.H3("Duplicated rows"),
                            dbc.Table.from_dataframe(self.explainer.Dataset[self.explainer.Dataset.duplicated()], striped=True, bordered=True, hover=True),
                    #dbc.Table(hyper_importance_table_header + hyper_importance_table_body, bordered=False),
                    #className="p-5"
                    #self.shap_dependence.layout()
                ])), html.Br(),
                ]),])


            
            
            
            #--------------------------------------------------------------------------------------------------------

class DataProfilingmed(ExplainerComponent):
    def __init__(self, explainer, title="Provided Data", name=None, **kwargs):

        super().__init__(explainer, title, name)
        self.scatter = ScatterComponent(explainer, name=self.name+"1", **kwargs)
        self.samples = SamplesComponent(explainer, name=self.name+"1", **kwargs)
        self.missingVal = MissComponent(explainer, name=self.name+"1", **kwargs)
        self.duplicatedRows = duplicatedComponent(explainer, name=self.name+"1", **kwargs)


    def layout(self):
        tabs=[
                dbc.Tab(label="Overview", tab_id="Overview"),
                dbc.Tab(label="Samples", tab_id="Samples"),
            ]
        
        # if self.explainer.Dataset.isnull().sum().sum() + self.explainer.Dataset.isna().sum().sum() >0:
        tabs.append(dbc.Tab(label="Missing values", tab_id="miss_vals"))
        if len(self.explainer.Dataset) -len(self.explainer.Dataset.drop_duplicates()) >0:
            tabs.append(dbc.Tab(label="Duplicate rows", tab_id="dup_rows"))                   
                    
        tabs.append(dbc.Tab(label="Data Correlation", tab_id="data_corr"))
        
        
        return html.Div([
            
            dbc.Card([
        dbc.CardHeader(
            dbc.Tabs(
                tabs,
                id="card-tabs",
                card=True,
                active_tab="Overview",
            )
        ),
        dbc.CardBody(html.P(id="card-content", className="card-text")),
        ])
        ]  )
    
    def component_callbacks(self, app):
        def DataOverview (ds):
            #dataset general info
            n_var = ds.shape[1]
            n_obs = ds.shape[0]
            n_missing = ds.isnull().sum().sum() + ds.isna().sum().sum()
            n_classes = ds.iloc[:, -1].nunique()
            dup_rows =len(ds)-len(ds.drop_duplicates())
            #varibales (data type) infos
            Numeric = ds.select_dtypes(include='number').shape[1]
            Categorical = ds.select_dtypes(include='object').shape[1]
            Boolean = ds.select_dtypes(include='bool').shape[1]
            Date = ds.select_dtypes(include='datetime64').shape[1]
            Unsupported = 0            
            global dsInfo, varInfo # les variable dsInfo et varInfo sont utilisé pour les informations du dataframe 
            dsInfo=[n_var, n_obs, n_classes,n_missing,  dup_rows]
            varInfo=[Numeric, Categorical, Boolean, Date, Unsupported]
            return dsInfo,varInfo


        
        dsInfo,varInfo = DataOverview(self.explainer.Dataset)
        
        @app.callback(Output("card-content", "children"), [Input("card-tabs", "active_tab")])




        def tab_content(active_tab):

            data = [{
            'values': [10,30,60],
            'type': 'pie'}]
            
            
            ds_info = pd.DataFrame({
                    " ": ["Number of variables", "Number of observations","Number of classes", "Missing cells", "Duplicate rows"],
                   "  ": dsInfo,})
            
            
            vr_type = pd.DataFrame({
                    " ": ["Numeric", "Categorical", "Boolean", "Date", "Unsupported"],
                   "  ": varInfo,})
            

            
            if active_tab=="Overview":
                return html.Div([
                    dbc.Row([     

                         dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader([

                                                html.H4("Dataset info", className="card-title"),
                                                html.Div([dbc.Badge("Warning", color="warning",className="ml-1")], style={"float": "right"}),

                                        ])  ,
                                    #tableau pour les info du dataset
                                    dbc.CardBody([
                                        dbc.Table.from_dataframe(ds_info, striped=True, bordered=True, hover=True,id='table'),

                                        #graphique de distribution des classe
                                        dcc.Graph(
                                            id='graph',
                                            figure={ 'data': data ,"layout": {"title": "Classes distribution","height": 400,  # px
                                            }, }
                                        #    }, }
                                        ), 

                                            ],id='card')
                                    ])
                                ]),
                           
                        
            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader([html.H4("Data type", className="card-title"),
                                                    ]),
                                    dbc.CardBody([#tableau pour le type des colonnes
                                    dbc.Table.from_dataframe(vr_type, striped=True, bordered=True, hover=True),
                                                
                                       
                                        html.Div([#les différent warning
                                    html.H4("Warnings"),
                                    dbc.Alert([
                                        "Dataset has ",
                        html.A(dsInfo[4], href="#", className="alert-link"),
                        html.A("("+str(round((dsInfo[4]*100)/dsInfo[1],2))+")%",  className="alert-link"),
                        " duplicate rows ",
                         html.Button('Drop dupplicate rows', id='submit-dupl', n_clicks=0, className="btn btn-outline-warning", style={"float": "right"}), 
                        dcc.Download(id="download-dupl"),],color="warning",style={"with": "500px"}),


                        dbc.Alert([
                        "Dataset has ",
                        html.A(dsInfo[3], href="#", className="alert-link"),
                        html.A("("+str(round((dsInfo[3]*100)/(dsInfo[0]*dsInfo[1]),2))+")%",),
                        " missing values ",
                        #html.Button('Compléter les valeurs manquantes', id='submit-val', n_clicks=0, className="btn btn-outline-warning"), 
                            #dcc.Download(id="download-text")
                            ], id="alert-auto",color="info",is_open=True, duration=1000), 

                                    dbc.Alert([
                        "Your ",
                        html.A('unbalanced dataset', href="#", className="alert-link"),
                        " will bias the prediction model towards the more common class!"
                        ],color="warning",is_open=True, duration=1000),

                        ],id="warnings") ])])
                                ]), 
                        
                        
              ]),  ])
            
    #------------------------------------------------------------------------------------------------------------------
            # classe= bf.iloc[:,-1].unique()
            if active_tab=="Samples":
                return html.Div([dbc.Row([dbc.Col([ self.samples.layout() ]) ])])

                    
   #------------------------------------------------------------------------------------------------------------------          
            if active_tab=="data_corr":
                return html.Div([dbc.Row([dbc.Col([ self.scatter.layout() ]) ])])
        
   #------------------------------------------------------------------------------------------------------------------          
            if active_tab=="miss_vals":
                return html.Div([dbc.Row([dbc.Col([ self.missingVal.layout() ]) ])])        
        
    #------------------------------------------------------------------------------------------------------------------          
            if active_tab=="dup_rows":
                return html.Div([dbc.Row([dbc.Col([ self.duplicatedRows.layout() ]) ])])        
        
    #--------------------------------------------------------------------------------------------------------
    def to_html(self, state_dict=None, add_header=True):
        ds=self.explainer.Dataset
        n_var = ds.shape[1]
        n_obs = ds.shape[0]
        n_missing = ds.isnull().sum().sum() + ds.isna().sum().sum()
        n_classes = ds.iloc[:, -1].nunique()
        dup_rows =len(ds)-len(ds.drop_duplicates())
        #varibales (data type) infos
        Numeric = ds.select_dtypes(include='number').shape[1]
        Categorical = ds.select_dtypes(include='object').shape[1]
        Boolean = ds.select_dtypes(include='bool').shape[1]
        Date = ds.select_dtypes(include='datetime64').shape[1]
        Unsupported = 0            
        dsInfo=[n_var, n_obs, n_classes,n_missing,  dup_rows]
        varInfo=[Numeric, Categorical, Boolean, Date, Unsupported]
        ds_info = pd.DataFrame({
                    " ": ["Number of variables", "Number of observations","Number of classes", "Missing cells", "Duplicate rows"],
                   "  ": dsInfo,})
            
            
        vr_type = pd.DataFrame({
                    " ": ["Numeric", "Categorical", "Boolean", "Date", "Unsupported"],
                   "  ": varInfo,})    

 
        
        
        html0 = to_html.table_from_df(ds_info)
        html0 = to_html.card(html0, title="Dataset info")
        
        html1 = to_html.table_from_df(vr_type)
        html1 = to_html.card(html1, title="Data type")
        
        html2 = to_html.hide(self.scatter.to_html(), False)
        html2 = to_html.card(html2, title="Data correlation")
        
        html=to_html.card_rows([html0,html1],[html2])
        
        if add_header:
            return to_html.add_header(html)
        return html
    
    
    
class ExportDash(ExplainerComponent):
    def __init__(self, explainer, title="Data profiling med", name=None,
                    hide_title=True, hide_selector=True, 
                    hide_globalcutoff=False,
                    hide_modelsummary=False, hide_confusionmatrix=False,
                    hide_precision=False, hide_classification=False,
                    hide_rocauc=False, hide_prauc=False,
                    hide_liftcurve=False, hide_cumprecision=False,
                    pos_label=None,
                    bin_size=0.1, quantiles=10, cutoff=0.5, **kwargs):

        super().__init__(explainer, title, name)


    def layout(self):
        
        return dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
