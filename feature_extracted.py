# Data Analysis
import pandas as pd  # data processing
import numpy as np  # linear algebra
from tqdm import tqdm  # Instantly make your loops show a smart progress meter
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# "Simple trend feature: fit a linear regression and return the coefficient"
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


# based on https://www.kaggle.com/inversion/basic-feature-benchmark
# ****************************
# Additional features
# [1] Quantiles based on https://www.kaggle.com/andrekos/basic-feature-benchmark-with-quantiles
# [2] Absolute values and Trend features from: https://www.kaggle.com/jsaguiar/baseline-with-abs-and-trend-features
# [3] Rolling Quantiles from: https://www.kaggle.com/wimwim/rolling-quantiles
# [4] Additional features from: https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
# [5] Skewness and kurtusis from: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
# ****************************

rows = 150_000  # Length of test segments
segments = int(np.floor(train.shape[0] / rows))  # Number of segments in the train dataset

# New train features dataframe
train_columns = ['mean', 'std', 'max', 'min', 'kurtosis', 'skew', 'mean10', 'X_seg_sum',
                 'q001', 'q01', 'q05', 'q95', 'q99', 'q999', 'iqr', 'trend', 'abs_max',
                 'abs_mean', 'abs_std', 'abs_X_seg_sum', 'abs_trend', 'abs_median',
                 'abs_q95', 'abs_q99', 'F_test', 'p_test', 'mean_change_abs',
                 'mean_change_rate', 'mean_roll_std_10', 'std_roll_std_10',
                 'max_roll_std_10', 'min_roll_std_10', 'q01_roll_std_10',
                 'q05_roll_std_10', 'q95_roll_std_10', 'q99_roll_std_10',
                 'mean_change_abs_roll_std_10', 'abs_max_roll_std_10',
                 'mean_roll_mean_10', 'std_roll_mean_10', 'max_roll_mean_10',
                 'min_roll_mean_10', 'q01_roll_mean_10', 'q05_roll_mean_10',
                 'q95_roll_mean_10', 'q99_roll_mean_10', 'mean_change_abs_roll_mean_10',
                 'mean_change_rate_roll_mean_10', 'abs_max_roll_mean_10',
                 'mean_roll_std_100', 'std_roll_std_100', 'max_roll_std_100',
                 'min_roll_std_100', 'q01_roll_std_100', 'q05_roll_std_100',
                 'q95_roll_std_100', 'q99_roll_std_100', 'mean_change_abs_roll_std_100', 'abs_max_roll_std_100',
                 'mean_roll_mean_100', 'std_roll_mean_100', 'max_roll_mean_100',
                 'min_roll_mean_100', 'q01_roll_mean_100', 'q05_roll_mean_100',
                 'q95_roll_mean_100', 'q99_roll_mean_100',
                 'mean_change_abs_roll_mean_100', 'mean_change_rate_roll_mean_100',
                 'abs_max_roll_mean_100',
                 'mean_roll_std_1000', 'std_roll_std_1000', 'max_roll_std_1000',
                 'min_roll_std_1000', 'q01_roll_std_1000', 'q05_roll_std_1000',
                 'q95_roll_std_1000', 'q99_roll_std_1000', 'mean_change_abs_roll_std_1000', 'abs_max_roll_std_100',
                 'mean_roll_mean_1000', 'std_roll_mean_1000', 'max_roll_mean_1000',
                 'min_roll_mean_1000', 'q01_roll_mean_1000', 'q05_roll_mean_1000',
                 'q95_roll_mean_1000', 'q99_roll_mean_1000',
                 'mean_change_abs_roll_mean_1000', 'mean_change_rate_roll_mean_1000',
                 'abs_max_roll_mean_1000']

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=train_columns)
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])


# Function to create Features
def create_features(segment, DF_seg, DF_X_Output):
    x = DF_seg['acoustic_data']
    DF_X_Output.loc[segment, 'mean'] = x.mean()
    DF_X_Output.loc[segment, 'std'] = x.std()
    DF_X_Output.loc[segment, 'max'] = x.max()
    DF_X_Output.loc[segment, 'min'] = x.min()
    DF_X_Output.loc[segment, 'kurtosis'] = x.kurtosis()  # [5]
    DF_X_Output.loc[segment, 'skew'] = x.skew()  # [5]
    # Trimmed mean, which excludes the outliers, of an array, in this case excludes 10% at both ends
    DF_X_Output.loc[segment, 'mean10'] = stats.trim_mean(x, 0.1)  # [2]
    DF_X_Output.loc[segment, 'X_seg_sum'] = x.sum()

    # Quantile
    DF_X_Output.loc[segment, 'q001'] = np.quantile(x, 0.001)  # [2]
    DF_X_Output.loc[segment, 'q01'] = np.quantile(x, 0.01)  # [1]
    DF_X_Output.loc[segment, 'q05'] = np.quantile(x, 0.05)  # [1]
    DF_X_Output.loc[segment, 'q95'] = np.quantile(x, 0.95)  # [1]
    DF_X_Output.loc[segment, 'q99'] = np.quantile(x, 0.99)  # [1]
    DF_X_Output.loc[segment, 'q999'] = np.quantile(x, 0.999)  # [2]
    # Interquartile range IQR: The IQR describes the middle 50% of values when ordered from lowest to highest.
    # IQR = q75 - q25
    DF_X_Output.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))  # [2]

    # Trends
    DF_X_Output.loc[segment, 'trend'] = add_trend_feature(x)  # [2]

    # Absolut Values
    DF_X_Output.loc[segment, 'abs_max'] = np.abs(x).max()  # [2]
    DF_X_Output.loc[segment, 'abs_mean'] = np.abs(x).mean()  # [2]
    DF_X_Output.loc[segment, 'abs_std'] = np.abs(x).std()  # [2]
    DF_X_Output.loc[segment, 'abs_X_seg_sum'] = np.abs(x).sum()
    DF_X_Output.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)  # [2]
    DF_X_Output.loc[segment, 'abs_median'] = np.median(np.abs(x))  # [3]
    DF_X_Output.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)  # [3]
    DF_X_Output.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)  # [3]

    # Change/diff in acoustic data within a segment [3]
    # Divide the segment in groups of 30000 sample as and do a oneway anova test
    DF_X_Output.loc[segment, 'F_test'], DF_X_Output.loc[segment, 'p_test'] = stats.f_oneway(x[:30000], x[30000:60000],
                                                                                            x[60000:90000],
                                                                                            x[90000:120000], x[120000:])
    DF_X_Output.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
    DF_X_Output.loc[segment, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    # Rolling features [3], and [4] added 1000 windows
    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values

        DF_X_Output.loc[segment, 'mean_roll_std_' + str(windows)] = x_roll_std.mean()
        DF_X_Output.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        DF_X_Output.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        DF_X_Output.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        DF_X_Output.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        DF_X_Output.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        DF_X_Output.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        DF_X_Output.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        DF_X_Output.loc[segment, 'mean_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        DF_X_Output.loc[segment, 'mean_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        DF_X_Output.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        DF_X_Output.loc[segment, 'mean_roll_mean_' + str(windows)] = x_roll_mean.mean()
        DF_X_Output.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        DF_X_Output.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        DF_X_Output.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        DF_X_Output.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        DF_X_Output.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        DF_X_Output.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        DF_X_Output.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        DF_X_Output.loc[segment, 'mean_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        DF_X_Output.loc[segment, 'mean_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        DF_X_Output.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()


# Create new features for the training dataset
for segment_id in tqdm(range(segments)):
    seg = train.iloc[segment_id * rows:segment_id * rows + rows]
    create_features(segment_id, seg, X_train)
    y_train.loc[segment_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

lm = LinearRegression()

# Train lm using the training data.
lm.fit(X_train_scaled, y_train)
