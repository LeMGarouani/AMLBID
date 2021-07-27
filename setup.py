from setuptools import setup, find_packages

setup(
    name='AMLBID',
    version='0.2',
    description='Transparent and Auto-explainable AutoML',
    long_description="""

AMLBID is a Python-Package representing a meta learning-based framework for automating the process of algorithm selection, and hyper-parameter tuning in supervised machine learning. Being meta-learning based, the framework is able to simulate the role of the machine learning expert as a decision support system. In particular, AMLBID is considered the first complete, transparent and auto-explainable AutoML system for recommending the most adequate ML configuration for a problem at hand, and explain the rationale behind the recommendation and analyzing the predictive results in an interpretable and faithful manner through an interactive multiviews artifact.

AMLBID is an interactive and user-guided framework for improving the utility and usability of the AutoML process with the following main features:

- The framework provides end-users (Industry 4.0 actors & Researchers) with a user-friendly control panel that allows nontechnical users and domain experts (e.g., physicians, researchers) to overcome machine-learning predictive models building and configuring process challenges according to their own preferences.

- The first framework system that automate machine-learning predictive models building and configuration for big industrial data. The tool will enable manufacturing actors and researchers to rapidly ask a series of what-if scenarios when probing opportunities to use predictive models to improve outcomes and reduce costs for various tasks as well as the need of classical collaborations.

- The framework is equipped with a recommendation engine, that provide suggestion of the most appropriate pipelines (classifiers with their hyperparameters configuration) through the use of a collaborative knowledge-base that grows by time as more users are using our tool.

- AMLBID will automate the most tedious part of machine learning by intelligently exploring more than 3.000.000 possible pipelines to find the best one for your data in a negligible amount of time and without need to a strong computational budget.

- Automatically select ML algorithms and hyperparameters configurations for a given machine-learning problem more quickly than current methods with a computational complexity near O(1).

- The first system that automatically and efficiently select ML pipeline and providing the rational of the provided suggestion. Existing AutoML tools cannot do this.

- The framework is equipped with an explanation artifact which allows the end-user to understand and diagnose the design of the returned machine learning models using various explanation techniques. In particular, the explanation artifact allows the end-user to:

+ Investigate of the reasoning behind the AutoML recommendation generation process.
Explore the predictions of any recommendation in a faithful way, through linked visual summary - textual informations for a higher trust.
+ Provide a multi-level interactive visualization artifact that facilitate the models workings and performance inspection to address the “black-box trusting”.

- Provide a guidance, when AutoML returns unsatisfying results, to improve to predictive performances by assessing the importance of an algorithm hyperparameters.

A deployed example can be found at https://colab.research.google.com/drive/1zpMdccwRsoWe8dmksp_awY5qBgkVwsHd?usp=sharing
""",
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(),
    package_dir={'AMLBID': 'AMLBID'},  
    package_data={
        'AMLBID': ['Explainer/assets/*', 'Recommender/builtins/KnowledgeBase/*'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Framework :: Dash",
        "Framework :: Flask",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    install_requires=['dash>=1.20', 'dash-bootstrap-components', 'jupyter_dash', 'dash-auth',
                    'dtreeviz>=1.3', 'numpy', 'pandas>=1.1', 'scikit-learn', 
                    'shap>=0.37', 'joblib', 'oyaml', 'click', 'waitress',
                    'flask_simplelogin', 'scikit-learn', 'xgboost', 'termcolor', 'pdpbox', 'shortuuid'],
    python_requires='>=3.6',
    author='Moncef GAROUANI',
    author_email='mgarouani@gmail.com',
    keywords=['machine learning', 'Automated machine learning', 'explainability'],
    url='https://github.com/LeMGarouani/AMLBID',
    project_urls={
        "Github page": "https://github.com/LeMGarouani/AMLBID",
        "Documentation": "https://github.com/LeMGarouani/AMLBID",
    },
)
