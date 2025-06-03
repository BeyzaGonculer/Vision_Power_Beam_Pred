## 📁 Dataset

To run this project, you must first download the **Scenario5** dataset manually.  
Once downloaded, place the dataset into the root directory of the project under a folder named `dataset`.

Your directory structure should look like this:



beam_prediction_project/
├── dataset/
│ └── unit1/
│ └── [image files...]
├── models/
├── src/
├── README.md
├── ...



> ⚠️ **Note:** The dataset is not included in this repository due to size limitations. Please download it manually.

---

## 🧠 Model Training

Each model should be trained, validated, and tested **independently** using the corresponding CSV files:

- `scenario5_dev_train.csv`
- `scenario5_dev_val.csv`
- `scenario5_dev_test.csv`

The training pipeline saves the resulting models as `.pth` files under the `models/` directory (or your specified output path).

> ✅ Make sure you follow the same data split and naming conventions for consistent evaluation.



