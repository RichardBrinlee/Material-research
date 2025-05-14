import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.nn import GATConv

# Define the Graph Neural Network
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.mlp(x)
        return x

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphAttentionNetwork, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('atomic.csv')

    # Define elements
    elements = ['Mo', 'Nb', 'Ta', 'W', 'V']

    # Parse the Element_Combination
    def parse_combination(combination):
        element_dict = {element: 0.0 for element in elements}
        parts = re.findall(r'([A-Z][a-z]*)(\d*\.\d+|\d+)', combination)
        for element, percentage in parts:
            element_dict[element] = float(percentage)
        return [element_dict[element] for element in elements]

    # Create node features
    nodes = [parse_combination(combination) for combination in data['Element_Combination']]
    nodes = torch.tensor(nodes, dtype=torch.float)

    # Normalize node features
    scaler_x = StandardScaler()
    nodes = scaler_x.fit_transform(nodes)
    nodes = torch.tensor(nodes, dtype=torch.float)

    # Create the target values (USFE and Lattice_Parameter)
    y = data[['USFE', 'Lattice_Parameter']].values

    # Normalize target values
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
    y = torch.tensor(y, dtype=torch.float)

    # Create graphs with edges based on a distance threshold
    distance_threshold = 1.0
    graphs = []
    for i in range(nodes.shape[0]):
        # Define the nodes for the current graph
        current_node = nodes[i].unsqueeze(0)  # Single node graph

        # Calculate distances between the current node and all other nodes
        distances = euclidean_distances(current_node, nodes)

        # Create edges based on the distance threshold
        edge_indices = torch.nonzero(torch.tensor(distances < distance_threshold, dtype=torch.bool), as_tuple=False).t()
        edge_weights = torch.tensor(distances[distances < distance_threshold], dtype=torch.float)

        # Ensure edge indices are valid for the current graph
        edge_indices = edge_indices[:, edge_indices[1] < current_node.shape[0]]

        # Add target values (y) to the graph
        target = y[i].unsqueeze(0)  # Target values for this graph

        # Create a graph
        graph = Data(x=current_node, edge_index=edge_indices, edge_attr=edge_weights, y=target)
        graphs.append(graph)

    print(f"Created {len(graphs)} graphs.")

    # Split data into training and testing sets
    train_idx, test_idx = train_test_split(range(len(graphs)), test_size=0.20, random_state=42)
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)

    # Initialize the model
    input_dim = nodes.shape[1]
    hidden_dim = 128
    output_dim = y.shape[1]
    # Initialize model, optimizer, and loss function
    model = GraphAttentionNetwork(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    '''
    # Training loop with adjusted loss weights
    def train():
        model.train()
        total_loss_usfe = 0
        total_loss_lattice = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss_usfe = criterion(out[:, 0], batch.y[:, 0])
            loss_lattice = criterion(out[:, 1], batch.y[:, 1])
            # Adjust the weights for the losses
            loss = 0.85 * loss_usfe + 0.15 * loss_lattice
            loss.backward()
            optimizer.step()
            total_loss_usfe += loss_usfe.item()
            total_loss_lattice += loss_lattice.item()
        avg_loss_usfe = total_loss_usfe / len(train_loader)
        avg_loss_lattice = total_loss_lattice / len(train_loader)
        return avg_loss_usfe, avg_loss_lattice
    '''
    def train():
        model.train()
        total_loss_usfe = 0
        total_loss_lattice = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            
            # Compute individual losses
            loss_usfe = criterion(out[:, 0], batch.y[:, 0])
            loss_lattice = criterion(out[:, 1], batch.y[:, 1])
            
            # Dynamically adjust weights based on the inverse of the current losses
            weight_usfe = 1 / (loss_usfe.item() + 1e-8)  # Add small value to avoid division by zero
            weight_lattice = 1 / (loss_lattice.item() + 1e-8)
            total_weight = weight_usfe + weight_lattice
            
            # Normalize weights
            weight_usfe /= total_weight
            weight_lattice /= total_weight
            
            # Compute the weighted loss
            loss = weight_usfe * loss_usfe + weight_lattice * loss_lattice
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            total_loss_usfe += loss_usfe.item()
            total_loss_lattice += loss_lattice.item()
        
        # Compute average losses for the epoch
        avg_loss_usfe = total_loss_usfe / len(train_loader)
        avg_loss_lattice = total_loss_lattice / len(train_loader)
        return avg_loss_usfe, avg_loss_lattice
    
    # Call the training loop
    for epoch in range(300):
        loss = train()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Evaluation with L2 regularization
    model.eval()
    predictions = []
    with torch.no_grad():
        total_loss_usfe = 0
        total_loss_lattice = 0
        l2_lambda = 5e-4  # L2 regularization weight
        for batch in test_loader:
            out = model(batch.x, batch.edge_index)
            loss_usfe = criterion(out[:, 0], batch.y[:, 0])
            loss_lattice = criterion(out[:, 1], batch.y[:, 1])
            
            # Add L2 regularization
            l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
            loss_usfe += l2_lambda * l2_reg
            loss_lattice += l2_lambda * l2_reg
            
            total_loss_usfe += loss_usfe.item()
            total_loss_lattice += loss_lattice.item()
            predictions.append(out)
        avg_loss_usfe = total_loss_usfe / len(test_loader)
        avg_loss_lattice = total_loss_lattice / len(test_loader)
        mse_loss = (avg_loss_usfe, avg_loss_lattice)
        print(f'Mean Squared Error with L2 Regularization: {mse_loss}')

    # Print predictions and actual values
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat([batch.y for batch in test_loader], dim=0)
    print("Predictions (first 5 rows):\n", scaler_y.inverse_transform(predictions[:5].detach().numpy()))
    print("Actual Values (first 5 rows):\n", scaler_y.inverse_transform(actuals[:5].detach().numpy()))

    # Save to CSV with Element_Combination and Percent Difference
    element_combinations = data['Element_Combination'].iloc[test_idx].reset_index(drop=True)
    actuals_np = scaler_y.inverse_transform(actuals.detach().numpy())
    predictions_np = scaler_y.inverse_transform(predictions.detach().numpy())

    percent_diff_usfe = 100 * abs(predictions_np[:, 0] - actuals_np[:, 0]) / actuals_np[:, 0]
    percent_diff_lattice = 100 * abs(predictions_np[:, 1] - actuals_np[:, 1]) / actuals_np[:, 1]

    test_data_df = pd.DataFrame({
        'Element_Combination': element_combinations,
        'USFE': actuals_np[:, 0],
        'Predicted_USFE': predictions_np[:, 0],
        'Percent_Difference_USFE': percent_diff_usfe,
        'Lattice_Parameter': actuals_np[:, 1],
        'Predicted_Lattice_Parameter': predictions_np[:, 1],
        'Percent_Difference_Lattice_Parameter': percent_diff_lattice
    })
    test_data_df.to_csv('predictions.csv', index=False)

    # print the average percent differences
    avg_percent_diff_usfe = percent_diff_usfe.mean()
    avg_percent_diff_lattice = percent_diff_lattice.mean()
    print(f'Average Percent Difference USFE: {avg_percent_diff_usfe:.2f}%')
    print(f'Average Percent Difference Lattice Parameter: {avg_percent_diff_lattice:.2f}%')

    # print the max percent differences
    max_percent_diff_usfe = percent_diff_usfe.max()
    max_percent_diff_lattice = percent_diff_lattice.max()
    print(f'Max Percent Difference USFE: {max_percent_diff_usfe:.2f}%')
    print(f'Max Percent Difference Lattice Parameter: {max_percent_diff_lattice:.2f}%')

    import matplotlib.pyplot as plt

    # Prepare data for visualization
    predictions_np = scaler_y.inverse_transform(predictions.detach().numpy())
    actuals_np = scaler_y.inverse_transform(actuals.detach().numpy())

    test_data_df = pd.DataFrame({
        'USFE': actuals_np[:, 0],
        'Predicted_USFE': predictions_np[:, 0],
        'Lattice_Parameter': actuals_np[:, 1],
        'Predicted_Lattice_Parameter': predictions_np[:, 1]
    })

    # Visualize predictions vs actual values
    plt.figure(figsize=(12, 6))

    # USFE
    plt.subplot(1, 2, 1)
    plt.scatter(test_data_df['USFE'], test_data_df['Predicted_USFE'], alpha=0.5, label='Predicted', color='red')
    plt.plot(test_data_df['USFE'], test_data_df['USFE'], label='Actual', color='blue')  # Line for actual values
    plt.xlabel('Actual USFE')
    plt.ylabel('Predicted USFE')
    plt.title('USFE: Actual vs Predicted')
    plt.legend()

    # Lattice Parameter
    plt.subplot(1, 2, 2)
    plt.scatter(test_data_df['Lattice_Parameter'], test_data_df['Predicted_Lattice_Parameter'], alpha=0.5, label='Predicted', color='red')
    plt.plot(test_data_df['Lattice_Parameter'], test_data_df['Lattice_Parameter'], label='Actual', color='blue')  # Line for actual values
    plt.xlabel('Actual Lattice Parameter')
    plt.ylabel('Predicted Lattice Parameter')
    plt.title('Lattice Parameter: Actual vs Predicted')
    plt.legend()

    plt.tight_layout()
    plt.show()