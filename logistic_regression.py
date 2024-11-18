import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

plt.switch_backend('Agg')

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                   [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1) and shift it
    X2 = np.random.multivariate_normal(mean=[1 + distance, 1 - distance], cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2



def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}

    n_samples = step_num
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(20, n_rows * 10))

    for i, distance in enumerate(shift_distances, 1):
        # Generate clusters and fit the logistic regression model
        X, y = generate_ellipsoid_clusters(distance=distance)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)

        # Calculate slope, intercept, and margin width
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        margin_width = 2 / np.linalg.norm([beta1, beta2])

        # Logistic loss
        probabilities = model.predict_proba(X)
        loss = log_loss(y, probabilities)

        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope_list.append(slope)
        intercept_list.append(intercept)
        loss_list.append(loss)
        margin_widths.append(margin_width)

        # Prepare for plotting
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Plotting the data points
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label="Class 0", s=10)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label="Class 1", s=10)

        # Plot decision boundary
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, c='black', linestyle='--', label="Decision Boundary")

        # Plot margin lines
        margin_offset = margin_width / 2 * np.array([-beta2, beta1]) / np.linalg.norm([beta1, beta2])
        margin_intercept1 = intercept - margin_offset[1]
        margin_intercept2 = intercept + margin_offset[1]
        plt.plot(
            x_vals, slope * x_vals + margin_intercept1, "k:", alpha=0.8, label="Margin Line"
        )
        plt.plot(
            x_vals, slope * x_vals + margin_intercept2, "k:", alpha=0.8
        )

        # Plot probability contours (confidence regions)
        contour_levels = [0.7, 0.8, 0.9]  # Example confidence levels
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=["red"], alpha=alpha)
            plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=["blue"], alpha=alpha)

        # Add logistic regression equation and margin width as text
        equation = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        plt.text(
            0.05, 0.95, equation, transform=plt.gca().transAxes,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top'
        )
        plt.text(
            0.05, 0.85, f"Margin Width: {margin_width:.2f}", transform=plt.gca().transAxes,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top'
        )

        # Format the plot
        plt.title(f"Shift Distance = {distance:.2f}", fontsize=16)
        plt.xlabel("x1")
        plt.ylabel("x2")
        pltmin = min(x_min, y_min)
        pltmax = min(x_max, y_max)
        plt.axis("tight")
        # Save the sample data for debugging or further analysis
        sample_data[distance] = (X, y, model, beta0, beta1, beta2)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    plt.close()
    # Plot parameters and metrics
    plt.figure(figsize=(18, 15))

    # Beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, label="Beta0")
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, label="Beta1")
    plt.title("Shift Distance vs Beta1")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, label="Beta2")
    plt.title("Shift Distance vs Beta2")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Slope
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, label="Slope")
    plt.title("Shift Distance vs Slope")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    # Intercept
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, label="Intercept")
    plt.title("Shift Distance vs Intercept")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    # Logistic Loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, label="Logistic Loss")
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Margin Width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, label="Margin Width")
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
