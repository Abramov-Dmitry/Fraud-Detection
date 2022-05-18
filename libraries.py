import warnings
from collections import Counter
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, confusion_matrix, f1_score
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     RandomizedSearchCV)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
