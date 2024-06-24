![https://pypi.python.org/pypi/explainerdashboard/](https://img.shields.io/pypi/v/AMLBID.svg)

# Transparent and Auto-explainable AutoML
<p align="center">
  <img src="https://github.com/LeMGarouani/AMLBID/blob/main/media/Framework.png" width="100%" />
</p>

---
# AMLBID
AMLBID stands for **A**utomating **M**achine-**L**earning model selection and configuration with **B**ig **I**ndustrial **D**ata.

Curently, **AMLBID** is a Python-Package representing a **meta learning**-based framework for automating the process of algorithm selection, and hyper-parameter tuning in supervised machine learning. Being meta-learning based, the framework is able to simulate the role of the machine learning expert as a **decision support system**. In particular, **AMLBID** is considered the first **complete**, **transparent** and **auto-explainable** AutoML system for **recommending** the most adequate ML configuration for a problem at hand, and **explain** the rationale behind the recommendation and analyzing the predictive results in an interpretable and faithful manner through an **interactive multiviews artifact**.  

**AMLBID** is an interactive and user-guided framework for improving the utility and usability of the *AutoML* process with the following main features:
 
* The framework provides  end-users *(Industry 4.0 actors & Researchers)* with a user-friendly *control panel* that allows nontechnical users and domain experts (e.g., physicians, researchers) to overcome machine-learning predictive models building and configuring process challenges according to their own preferences.

* The first framework system that automate machine-learning predictive models building and configuration for big industrial data.

* The framework is equipped with a **recommendation engine**, that provide suggestion of the most appropriate pipelines *(classifiers with their hyperparameters configuration)* through the use of a collaborative **knowledge-base** that grows by time as more users are using our tool.

* AMLBID will automate the most tedious part of machine learning by intelligently exploring more than **4.000.000** possible pipelines to find the best one for your data in a **negligible amount of time** and **without** need to a strong **computational budget**.

* Automatically select ML algorithms and hyperparameters configurations for a given machine-learning problem more quickly than current methods with a **computational complexity near O(1)**.

* Provide a multi-level interactive visualization artifact that facilitate the models workings and performance **inspection** to address the **“black-box trusting”**.

<p align="center">
<img alt="--" src="https://github.com/LeMGarouani/AMLBID/blob/main/media/AMLBID.png" width="80%" />
</p>

---
## Updates  ![https://upload.wikimedia.org/wikipedia/commons/archive/d/d4/20180501233355%21Software-update-available.svg](https://upload.wikimedia.org/wikipedia/commons/archive/d/d4/20180501233355%21Software-update-available.svg) 


 - A new data profiling tool that allows you to explore input data and automatically perform pre-processing tasks such as data imputation, normalization, encoding, and duplicate cleaning is added to the AMLBID explainer module.
 - You can **export** the generated dashboard as a dynamic HTML and PDF report file.
 - The new release is now compiled with the <a href = "https://pyinstaller.org/en/stable/">PyInstaller</a> for almost all well-known operating systems.  See [Packaging Your Python App Into Standalone Executables | PyInstaller](https://www.youtube.com/watch?v=s-lKHA9o_pY) for info on how to install AMLBID as a desktop standalone distribution.
---
## Usage
The `Framework` will help you with:
 - Explaining and understanding your data.
 - Automate the  Algorithm Selection and Hyper-Parameters tuning process.
 - Provide reports from analysis with details about all models (Atomatic-Explanation).
 - Interactively inspect the inner workings of the models without having to depend on a data scientist to generate tables and plots.
 - Provide a guidance, when AutoML returns unsatisfying results, to improve to predictive performances.
 - Increase the transparency, controllability, and the acceptance of AutoML.

It has two built-in modes of work:
 - **`Recommender`** mode, for recommending and building highly-tuned ML pipelines to use in production.
 - **`Recommender_Explainer`** mode, which allow users to inspect the recommended model's inner working and decision’s generation process, with many explanations levels, like feature importances, feature contributions to individual predictions, "what if" analysis, SHAP (interaction) values, visualisation of individual decision trees, Hyperameters inportance and correlation etc.

Curently,supports 08 <a href = "https://scikit-learn.org/stable/"> Scikit-Learn </a> classification algorithms, `AdaBoost`, `Support Vector Classifier`, `Extra Trees`, `Gradient Boosting`, `Decision Tree`, `Logistic Regression`, `Random Forest`, and `Stochastic Gradient Descent Classifier`. 

## Installation
AMLBID is built on top of several existing Python libraries, including:
* numpy
* shap
* jupyter_dash
* xgboost
* [dtreeviz](https://github.com/parrt/dtreeviz) **Windows users:** *See https://github.com/parrt/dtreeviz for info on how to properly install dtreeviz*.
* ...

Most of the necessary Python packages can be installed via the PyPi packages index or Anaconda Python distribution.

```python
# Install additional Python requirements
pip install -r requirements.txt
```
Finally to install AMLBID itself along with required dependencies, run the following command:
```python
# Install additional Python requirements
pip install AMLBID
```
## Examples of use
A working example is deployed in: [AMLBID](https://colab.research.google.com/drive/1zpMdccwRsoWe8dmksp_awY5qBgkVwsHd?usp=sharing)

#### <ins>Mode `Recommender`</ins>:

Below is a minimal working example of the `Recommender`mode .

```python
from AMLBID.Recommender import AMLBID_Recommender
from AMLBID.loader import *

#load dataset
Data,X_train,Y_train,X_test,Y_test=load_data("TestData.csv")

#Generate the optimal configuration according to a desired predictive metric
AMLBID=AMLBID_Recommender.recommend(Data, metric="Accuracy", mode="Recommender")
AMLBID.fit(X_train, Y_train)
print("obtained score:", AMLBID.score(X_test, Y_test))
```
The corresponding Python code of the recommended pipeline should be exported to the `Recommended_pipeline.py` file and look similar to the following:<br/>
*Note that the packages import code is generated automatically and dynamically according to the recommended ML pipeline.*

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("Evaluation/Dataset.csv")

X = data.drop('class', axis=1)
Y = data['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model= DecisionTreeClassifier(criterion='entropy', max_features=0.5672564318672457,
                       min_samples_leaf=5, min_samples_split=20)
                       
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
score = model.score(X_test, Y_test)

print(classification_report(Y_test, Y_pred))
print(' Pipeline test accuracy:  %.3f' % score)
```
---

#### <ins>Mode `Recommender_Explainer`</ins>:

Below is a minimal working example of the `Recommender_Explainer` mode.
```python
from AMLBID.Recommender import AMLBID_Recommender
from AMLBID.Explainer import AMLBID_Explainer
from AMLBID.loader import *

#load dataset
Data,X_train,Y_train,X_test,Y_test=load_data("TestData.csv")

#Generate the optimal configurations according to a desired predictive metric
AMLBID,Config=AMLBID_Recommender.recommend(Data, metric="Accuracy", mode="Recommender_Explainer")
AMLBID.fit(X_train, Y_train)

#Generate the interactive explanatory dash
Explainer = AMLBID_Explainer.explain(AMLBID,Config, X_test, Y_test)
Explainer.run()
```
Demonstration of the explanatory artifact:

![https://github.com/LeMGarouani/AMLBID/blob/main/media/demo2.0_optimized.gif](https://github.com/LeMGarouani/AMLBID/blob/main/media/demo2.0_optimized.gif)
---

## Citing AMLBID

If you use the AMLBID in a scientific publication, please consider citing at least one of the following papers:


* **Garouani, M., Ahmad, A., Bouneffa, M. et al. Towards big industrial data mining through explainable automated machine learning. Int J Adv Manuf Technol 120, 1169–1188 (2022). https://doi.org/10.1007/s00170-022-08761-9**

```bibtex
@article{Garouani2022a,
  doi = {10.1007/s00170-022-08761-9},
  url = {https://doi.org/10.1007/s00170-022-08761-9},
  year = {2022},
  month = feb,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {120},
  number = {1-2},
  pages = {1169--1188},
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Mohamed Hamlich and Gregory Bourguin and Arnaud Lewandowski},
  title = {Towards big industrial data mining through explainable automated machine learning},
  journal = {The International Journal of Advanced Manufacturing Technology}
}

```

* **Garouani, M., Ahmad, A., Bouneffa, M. et al. Using meta-learning for automated algorithms selection and configuration: an experimental framework for industrial big data. J Big Data 9, 57 (2022). https://doi.org/10.1186/s40537-022-00612-4**

```bibtex
@article{Garouani2022b,
  doi = {10.1186/s40537-022-00612-4},
  url = {https://doi.org/10.1186/s40537-022-00612-4},
  year = {2022},
  month = apr,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {9},
  number = {1},
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Mohamed Hamlich and Gregory Bourguin and Arnaud Lewandowski},
  title = {Using meta-learning for automated algorithms selection and configuration: an experimental framework for industrial big data},
  journal = {Journal of Big Data}
}

```

* **Garouani, M., Ahmad, A., Bouneffa, M. et al. Autoencoder-kNN meta-model based data characterization approach for an automated selection of AI algorithms. J Big Data 10, 14 (2023). https://doi.org/10.1186/s40537-023-00687-7**

```bibtex
@article{Garouani2023_jbd,
  doi = {10.1186/s40537-023-00687-7},
  url = {https://doi.org/10.1186/s40537-023-00687-7},
  year = {2023},
  month = feb,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {10},
  number = {14},
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Mohamed Hamlich},
  title = {Autoencoder-kNN meta-model based data characterization approach for an automated selection of AI algorithms},
  journal = {Journal of Big Data}
}

```

* **M. Garouani, A. Ahmad, M. Bouneffa, M. Hamlich, AMLBID: An auto-explained automated machine learning tool for big industrial data, SoftwareX 17 (2022) 100919. Doi: https://doi.org/10.1016/j.softx.2021.100919**

```bibtex
@article{Garouani2022,
  doi = {10.1016/j.softx.2021.100919},
  url = {https://doi.org/10.1016/j.softx.2021.100919},
  year = {2022},
  month = jan,
  publisher = {Elsevier {BV}},
  volume = {17},
  pages = {100919},
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Mohamed Hamlich},
  title = {{AMLBID}: An auto-explained Automated Machine Learning tool for Big Industrial Data},
  journal = {{SoftwareX}}
}
```

* **M. Garouani, A. Ahmad, M. Bouneffa, M. Hamlich, AMLBID 2.0: An auto-explained Automated Machine Learning tool for Big Industrial Data, SoftwareX 23 (2023) 100919. Doi: https://doi.org/10.1016/j.softx.2023.101444**

```bibtex
@article{Garouani2023_softx,
  doi = {10.1016/j.softx.2023.101444},
  title = {Version [2.0]- [AMLBID: An auto-explained Automated Machine Learning tool for Big Industrial Data]},
  journal = {SoftwareX},
  volume = {23},
  pages = {101444},
  year = {2023},
  issn = {2352-7110},
  author = {Moncef Garouani and Mourad Bouneffa and Adeel Ahmad and Mohamed Hamlich},
}
```

* **Garouani, M., Bouneffa, M. Automated machine learning hyperparameters tuning through meta-guided Bayesian optimization. Prog Artif Intell (2024). https://doi.org/10.1007/s13748-023-00311-y**

```bibtex
@article{Garouani2024_pai,
author = {Garouani, Moncef and Bouneffa, Mourad},
 doi = {10.1007/s13748-023-00311-y},
 issn = {2192-6360},
 journal = {Progress in Artificial Intelligence},
 month = {January},
 title = {Automated machine learning hyperparameters tuning through meta-guided Bayesian optimization},
 year = {2024}
}
```


* **Garouani, M., Bouneffa, M. (2023). Unlocking the Black Box: Towards Interactive Explainable Automated Machine Learning. In: Intelligent Data Engineering and Automated Learning – IDEAL 2023. IDEAL 2023. Lecture Notes in Computer Science, vol 14404. Springer, Cham. https://doi.org/10.1007/978-3-031-48232-8_42**

```bibtex
@InProceedings{garouani_IDEAL23,
author={Moncef Garouani and Mourad Bouneffa}
title={Unlocking the Black Box: Towards Interactive Explainable Automated Machine Learning},
booktitle={Intelligent Data Engineering and Automated Learning -- IDEAL 2023},
year={2023},
publisher={Springer Nature Switzerland},
pages={458--469},
doi={10.1007/978-3-031-48232-8_42},
isbn={978-3-031-48232-8}
}

```


* **Garouani, M., Ahmad, A., Bouneffa, M., Hamlich, M. (2022). Scalable Meta-Bayesian Based Hyperparameters Optimization for Machine Learning. In: Smart Applications and Data Analysis. SADASC 2022. Communications in Computer and Information Science, vol 1677. Springer, Cham. https://doi.org/10.1007/978-3-031-20490-6_14**

```bibtex
@inproceedings{GarouaniSADASC,
  doi = {10.1007/978-3-031-20490-6_14},
  year = {2022},
  publisher="Springer International Publishing",
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Mohamed Hamlich},
  title = {Scalable Meta-Bayesian Based Hyperparameters Optimization for Machine Learning},
  booktitle = {Smart Applications and Data Analysis}
}

```


* **Chaabi, M., Hamlich, M., Garouani, M. (2023). Evaluation of AutoML Tools for Manufacturing Applications. In: Azrar, L., et al. Advances in Integrated Design and Production II. CIP 2022. Lecture Notes in Mechanical Engineering. Springer, Cham. https://doi.org/10.1007/978-3-031-23615-0_33**

```bibtex
@InProceedings{10.1007/978-3-031-23615-0_33,
author="Chaabi, Meryem
and Hamlich, Mohamed
and Garouani, Moncef",
title="Evaluation of AutoML Tools for Manufacturing Applications",
booktitle="Advances in Integrated Design and Production II",
year="2023",
publisher="Springer International Publishing",
address="Cham",
pages="323--330",
isbn="978-3-031-23615-0"
}


```



* **Garouani, M.; Ahmad, A. and Bouneffa, M. (2023). Explaining Meta-Features Importance in Meta-Learning Through Shapley Values. In Proceedings of the 25th International Conference on Enterprise Information Systems - Volume 1: ICEIS; ISBN 978-989-758-648-4; ISSN 2184-4992, SciTePress, pages 591-598. DOI: 10.5220/0011986600003467**

```bibtex
@inproceedings{garouani_iceis23,
author={Moncef Garouani and Adeel Ahmad and Mourad Bouneffa.},
title={Explaining Meta-Features Importance in Meta-Learning Through Shapley Values},
booktitle={Proceedings of the 25th International Conference on Enterprise Information Systems - Volume 1: ICEIS,},
year={2023},
pages={591-598},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0011986600003467},
isbn={978-989-758-648-4},
issn={2184-4992},
}}

```

* **Garouani, M., Ahmad, A., Bouneffa, M., Hamlich, M., Bourguin, G., Lewandowski, A. (2022). Towards Meta-Learning Based Data Analytics to Better Assist the Domain Experts in Industry 4.0. In: Artificial Intelligence in Data and Big Data Processing. ICABDE 2021. Lecture Notes on Data Engineering and Communications Technologies, vol 124. Springer, Cham. https://doi.org/10.1007/978-3-030-97610-1_22**

```bibtex
@incollection{Garouani2022,
  doi = {10.1007/978-3-030-97610-1_22},
  url = {https://doi.org/10.1007/978-3-030-97610-1_22},
  year = {2022},
  publisher = {Springer International Publishing},
  pages = {265--277},
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Mohamed Hamlich and Gregory Bourguin and Arnaud Lewandowski},
  title = {Towards Meta-Learning Based Data Analytics to~Better Assist the~Domain Experts in~Industry 4.0},
  booktitle = {Artificial Intelligence in Data and Big Data Processing}
}

```

* **Garouani, M., Ahmad, A., Bouneffa, M., Hamlich, M., Bourguin, G., Lewandowski, A. (2022). Towards Meta-Learning Based Data Analytics to Better Assist the Domain Experts in Industry 4.0. In: Artificial Intelligence in Data and Big Data Processing. ICABDE 2021. Lecture Notes on Data Engineering and Communications Technologies, vol 124. Springer, Cham. https://doi.org/10.1007/978-3-030-97610-1_22**

```bibtex
@incollection{Garouani2022,
  doi = {10.1007/978-3-030-97610-1_22},
  url = {https://doi.org/10.1007/978-3-030-97610-1_22},
  year = {2022},
  publisher = {Springer International Publishing},
  pages = {265--277},
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Mohamed Hamlich and Gregory Bourguin and Arnaud Lewandowski},
  title = {Towards Meta-Learning Based Data Analytics to~Better Assist the~Domain Experts in~Industry 4.0},
  booktitle = {Artificial Intelligence in Data and Big Data Processing}
}
```


* **Garouani, M.; Ahmad, A.; Bouneffa, M.; Lewandowski, A.; Bourguin, G. and Hamlich, M. (2021). Towards the Automation of Industrial Data Science: A Meta-learning based Approach. In Proceedings of the 23rd International Conference on Enterprise Information Systems - Volume 1: ICEIS, 709-716. https://doi.org/10.5220/0010457107090716**

```bibtex
@inproceedings{Garouani2021,
  doi = {10.5220/0010457107090716},
  url = {https://doi.org/10.5220/0010457107090716},
  year = {2021},
  publisher = {{SCITEPRESS} - Science and Technology Publications},
  author = {Moncef Garouani and Adeel Ahmad and Mourad Bouneffa and Arnaud Lewandowski and Gregory Bourguin and Mohamed Hamlich},
  title = {Towards the Automation of Industrial Data Science: A Meta-learning based Approach},
  booktitle = {Proceedings of the 23rd International Conference on Enterprise Information Systems}
}

```


**AMLBID** was developed in the [IRIT Lab]([https://www.irit.fr/]), [LISIC Lab](https://www-lisic.univ-littoral.fr/) with funding from the [ULCO](https://www.univ-littoral.fr/), [HESTIM](https://www.hestim.ma/), and [CNRST](https://cnrst.ma/index.php/fr/).
