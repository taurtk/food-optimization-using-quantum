from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from pulser import Register, Sequence, Pulse
from pulser.devices import MockDevice
from pulser_simulation import QutipEmulator
import io

app = Flask(__name__)

class FoodWasteOptimizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.quantum_sequence = None
        self.register = None
        self.batch_size = 4  # Process 4 items at a time

    def preprocess_data(self, df):
        df = df.copy()
        df['Receiving_Date'] = pd.to_datetime(df['Receiving Date'])
        df['Expiration_Date'] = pd.to_datetime(df['Expiration Date'])
        df['Current_Date'] = pd.Timestamp.now()
        df['Remaining_Life'] = (df['Expiration_Date'] - df['Current_Date']).dt.days
        df['Shelf_Life_Total'] = df['Shelf Life (days)']
        df.loc[df['Shelf_Life_Total'] <= 0, 'Shelf_Life_Total'] = 1
        df['Life_Percentage'] = (df['Remaining_Life'] / df['Shelf_Life_Total'] * 100).clip(0, 100)
        df['Daily_Sales'] = df['Sales Velocity (units/day)']
        df['Current_Stock'] = df['Stock Level']
        df['Storage_Temp'] = df['Storage Temp (°C)']
        df['Current_Humidity'] = df['Humidity (%)']

        target_conditions = {
            'Meat': {'temp': 2, 'humidity': 85},
            'Dairy': {'temp': 4, 'humidity': 75},
            'Vegetables': {'temp': 8, 'humidity': 90},
            'Fruits': {'temp': 10, 'humidity': 85},
            'Beverages': {'temp': 15, 'humidity': 65},
            'Snacks': {'temp': 20, 'humidity': 60}
        }

        df['Target_Temp'] = df['Product Category'].map(lambda x: target_conditions.get(x, {'temp': 20})['temp'])
        df['Target_Humidity'] = df['Product Category'].map(lambda x: target_conditions.get(x, {'humidity': 60})['humidity'])
        df['Temperature_Risk'] = (abs(df['Storage_Temp'] - df['Target_Temp']) / 10).clip(0, 1)
        df['Humidity_Risk'] = (abs(df['Current_Humidity'] - df['Target_Humidity']) / 20).clip(0, 1)

        quality_map = {'Premium': 0.1, 'A': 0.2, 'B': 0.4, 'C': 0.6, 'Standard': 0.3}
        df['Quality_Risk'] = df['Quality Grade'].map(quality_map).fillna(0.3)
        df['Is_Season'] = df.apply(lambda row: row['Seasonal Indicator'] == 'All-Year' or row['Seasonal Indicator'] == datetime.now().strftime('%B'), axis=1)
        df['Season_Risk'] = np.where(df['Is_Season'], 0.2, 0.4)
        df['Promotion_Factor'] = df['Promotion Impact (%)'] / 100
        df['Lead_Time_Days'] = 3

        return df

    def create_quantum_waste_model(self, df):
        try:
            n_items = len(df)
            n_batches = (n_items + self.batch_size - 1) // self.batch_size
            all_results = []

            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_items)
                batch_df = df.iloc[start_idx:end_idx]

                features = np.column_stack((
                    batch_df['Temperature_Risk'],
                    batch_df['Humidity_Risk'],
                    batch_df['Quality_Risk'],
                    batch_df['Season_Risk']
                ))

                min_distance = 6.0
                features = features * min_distance

                grid_size = int(np.ceil(np.sqrt(len(features))))
                grid = np.zeros((len(features), 2))

                for i in range(len(features)):
                    row = i // grid_size
                    col = i % grid_size
                    grid[i] = [row * min_distance, col * min_distance]

                self.register = Register.from_coordinates(grid, prefix=f"batch_{batch}")
                self.quantum_sequence = Sequence(self.register, MockDevice)

                for i in range(len(batch_df)):
                    channel = f"waste_ch_{i}"
                    self.quantum_sequence.declare_channel(channel, "rydberg_global")

                    risk_score = (
                        0.3 * batch_df.iloc[i]['Temperature_Risk'] +
                        0.2 * batch_df.iloc[i]['Humidity_Risk'] +
                        0.2 * batch_df.iloc[i]['Quality_Risk'] +
                        0.2 * batch_df.iloc[i]['Season_Risk'] +
                        0.1 * (1 - batch_df.iloc[i]['Promotion_Factor'])
                    )

                    amplitude = np.pi * risk_score
                    duration = max(1, int(batch_df.iloc[i]['Life_Percentage']))

                    pulse = Pulse.ConstantPulse(duration, amplitude, 0, 0)
                    self.quantum_sequence.add(pulse, channel)

                try:
                    sim = QutipEmulator.from_sequence(
                        self.quantum_sequence,
                        sampling_rate=0.1
                    )
                    final_state = sim.run().get_final_state()
                    batch_results = np.real(final_state.diag())
                    all_results.extend(batch_results)
                except Exception as e:
                    batch_results = [
                        self._calculate_classical_risk(row)
                        for _, row in batch_df.iterrows()
                    ]
                    all_results.extend(batch_results)

            return np.array(all_results)

        except Exception as e:
            return np.array([
                self._calculate_classical_risk(row)
                for _, row in df.iterrows()
            ])

    def _calculate_classical_risk(self, row):
        return (
            0.3 * row['Temperature_Risk'] +
            0.2 * row['Humidity_Risk'] +
            0.2 * row['Quality_Risk'] +
            0.2 * row['Season_Risk'] +
            0.1 * (1 - row['Promotion_Factor'])
        )

    def optimize_inventory(self, df, quantum_results):
        if quantum_results is None:
            return df

        if np.isscalar(quantum_results):
            quantum_results = np.full(len(df), quantum_results)

        quantum_min = np.min(quantum_results)
        quantum_max = np.max(quantum_results)

        if quantum_min == quantum_max:
            waste_risks = np.full(len(df), 0.5)
        else:
            waste_risks = (quantum_results - quantum_min) / (quantum_max - quantum_min)

        df['Waste_Risk'] = waste_risks

        df['Optimal_Stock'] = df.apply(
            lambda row: self._calculate_optimal_stock(
                row['Daily_Sales'],
                row['Waste_Risk'],
                row['Remaining_Life'],
                row['Promotion_Factor']
            ), axis=1
        )

        df['Reorder_Point'] = df.apply(
            lambda row: self._calculate_reorder_point(
                row['Optimal_Stock'],
                row['Waste_Risk'],
                row['Lead_Time_Days'],
                row['Promotion_Factor']
            ), axis=1
        )

        return df

    def _calculate_optimal_stock(self, sales_volume, waste_risk, remaining_life, promotion_factor):
        base_stock = sales_volume * min(remaining_life, 7)
        promotion_adjustment = 1 + promotion_factor
        risk_adjustment = 1 - waste_risk
        return max(1, int(base_stock * risk_adjustment * promotion_adjustment))

    def _calculate_reorder_point(self, optimal_stock, waste_risk, lead_time, promotion_factor):
        safety_stock = optimal_stock * waste_risk * 0.5 * (1 + promotion_factor)
        return max(1, int(optimal_stock * (lead_time / 7) + safety_stock))

@app.route("/optimize", methods=['POST'])
def optimize_inventory():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        contents = file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        optimizer = FoodWasteOptimizer()

        processed_df = optimizer.preprocess_data(df)
        results = optimizer.create_quantum_waste_model(processed_df)

        if len(results) != len(processed_df):
            results = np.pad(results, (0, len(processed_df) - len(results)), constant_values=0.5)

        optimized_df = optimizer.optimize_inventory(processed_df, results)

        response_data = optimized_df.to_dict(orient="records")
        return jsonify({"data": response_data})

    except Exception as e:
        return jsonify({"error": f"Error optimizing inventory: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
