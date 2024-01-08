import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function to read data from the file located in the
# same directory as the code
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return np.array(data)

# Function to calculate the mean and required value X
def calculate_statistics(data):
    # Calculate the mean (approximated as  ̃W)
    mean_salary = np.mean(data)
    mean_salary = round(mean_salary,2)

    # Calculate the required value X: fraction of population
    # with salaries between  ̃W and 1.25  ̃W
    lower_bound = mean_salary
    upper_bound = 1.25 * mean_salary
    fraction_within_range = np.sum((data >= lower_bound) & (data <= upper_bound)) / len(data)
    fraction_within_range = round(fraction_within_range, 2)

    return mean_salary, fraction_within_range

# Function to plot the probability density function as a histogram and add a distribution curve
def plot_histogram_and_curve(data, mean_salary, fraction_within_range):
    plt.hist(data, bins=30, density=True, alpha=0.7, color='pink', label='Salary Distribution')

    # Fit a normal distribution to the data
    mu, std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Fit result (μ={mu:.2f}, σ={std:.2f})')

    # Plot vertical lines for mean and X
    plt.axvline(mean_salary, color='red', linestyle='dashed', linewidth=2,
                label='Mean Salary ( ̃W)='+str(mean_salary))
    plt.axvline(1.25 * mean_salary, color='green', linestyle='dashed',
                linewidth=2, label='1.25  ̃W'+str(fraction_within_range))

    # Add labels, title, and legend
    plt.xlabel('Annual Salary (Euros)')
    plt.ylabel('Probability Density')
    plt.title('Salary Distribution and Statistics with Normal Distribution Fit')
    plt.legend()


    # Show the plot
    plt.show()

# Main script
if __name__ == "__main__":
    # Read data from the file in the same directory
    file_path = 'data6-1.csv'  # Replace with the actual file name
    data = read_data(file_path)

    # Calculate mean and required value X
    mean_salary, fraction_within_range = calculate_statistics(data)
    print("mean salary = "+str(mean_salary)+" Euros")
    print("fraction within range="+str(fraction_within_range))

    # Plot histogram and statistics with normal distribution fit
    plot_histogram_and_curve(data, mean_salary, fraction_within_range)
