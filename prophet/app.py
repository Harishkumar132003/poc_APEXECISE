from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
retail = pd.read_csv("data/poc_retail.csv")
wholesale = pd.read_csv("data/poc_wholesale.csv")
stock = pd.read_csv("data/poc_stock_closing.csv")

# ---------------------------------------------------------
# FIX DATA TYPES
# ---------------------------------------------------------
retail["entity_code"] = retail["entity_code"].astype(str)
wholesale["from_entity_code"] = wholesale["from_entity_code"].astype(str)
wholesale["to_entity_code"] = wholesale["to_entity_code"].astype(str)
stock["entity_code"] = stock["entity_code"].astype(str)

retail["package_size"] = retail["package_size"].astype(str)
stock["package_size"] = stock["package_size"].astype(str)

# Fix dates
retail['bill_date'] = pd.to_datetime(retail['bill_date'], errors='coerce')
retail['ProductCategory'] = retail['brand_name'] + " | " + retail['package_size']


@app.route("/prophet/depot/predict", methods=["POST"])
def predict():
    req = request.json
    months = req.get("month", 2)
    from_month = req.get("from_month")    # NEW FIELD
    depotid = str(req.get("id"))

    if depotid is None:
        return jsonify({"error": "depotid is required"}), 400

    if from_month is None:
        return jsonify({"error": "from_month is required"}), 400

    # 1️⃣ RETAIL SHOPS UNDER DEPOT
    retail_shops = wholesale[wholesale["from_entity_code"] == depotid]["to_entity_code"].unique()

    if len(retail_shops) == 0:
        return jsonify({"error": "No retail shops found for this depot"}), 404

    # 2️⃣ FULL SALES DATA
    depot_sales_full = retail[retail["entity_code"].isin(retail_shops)]

    if depot_sales_full.empty:
        return jsonify({"error": "No retail sales found for this depot"}), 404

    # Determine prediction year dynamically (using latest year in data)
    year = depot_sales_full["bill_date"].dt.year.max()

    # ---------------------------------------------------------
    # 3️⃣ BUILD TRAINING WINDOW BASED ON from_month & months
    # ---------------------------------------------------------
    train_start_month = from_month - months
    train_end_month = from_month - 1

    train_start_date = datetime(year, train_start_month, 1)
    train_end_date = datetime(year, train_end_month, 1) + pd.offsets.MonthEnd(1)

    # Filter training period
    depot_sales_recent = depot_sales_full[
        (depot_sales_full["bill_date"] >= train_start_date) &
        (depot_sales_full["bill_date"] <= train_end_date)
    ]

    results = []

    # ALL SKUs handled by depot
    all_skus = depot_sales_full[["brand_name", "package_size"]].drop_duplicates().values.tolist()

    # ---------------------------------------------------------
    # 4️⃣ LOOP THROUGH ALL SKUs
    # ---------------------------------------------------------
    for brand, size in all_skus:

        sku_recent = depot_sales_recent[
            (depot_sales_recent["brand_name"] == brand) &
            (depot_sales_recent["package_size"] == size)
        ]

        sku_df = (
            sku_recent.groupby("bill_date")["sold_qty"]
            .sum()
            .reset_index()
            .rename(columns={"bill_date": "ds", "sold_qty": "y"})
        )

        # CASE A: Enough data -> run Prophet
        if len(sku_df) >= 5:
            model = Prophet()
            model.fit(sku_df)

            # 5️⃣ FORECAST NEXT MONTH → from_month
            predict_start = datetime(year, from_month, 1)
            predict_end = predict_start + pd.offsets.MonthEnd(1)

            future_dates = pd.date_range(start=predict_start, end=predict_end)
            future = pd.DataFrame({"ds": future_dates})

            forecast = model.predict(future)
            demand = float(forecast["yhat"].sum())

        else:
            # CASE B: No recent data -> demand = 0
            demand = 0.0

        # ---------------------------------------------------------
        # REMAINING STOCK
        # ---------------------------------------------------------
        depot_stock = stock[
            (stock["entity_code"] == depotid) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum()

        retail_stock_amt = stock[
            (stock["entity_code"].isin(retail_shops)) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum()

        remaining_stock = int(depot_stock + retail_stock_amt)

        # QUANTITY TO RAISE
        qty_to_raise = max(demand - remaining_stock, 0)
        quantitytoraise = int(round(qty_to_raise))
        rounddemand = int(round(demand))

        results.append({
            "brand": str(brand),
            "package_size": str(size),
            "remaining_at_depot": int(depot_stock),
             "remaining_at_retail": int(retail_stock_amt),
            "demand": int(rounddemand),
            "remaining_stock": remaining_stock,
            "quantitytoraise": quantitytoraise
        })

    return jsonify(results)

@app.route("/prophet/distillery/predict", methods=["POST"])
def predict_distillery():
    req = request.json
    months = req.get("month", 2)
    from_month = req.get("from_month")
    distillery_id = str(req.get("id"))

    if distillery_id is None:
        return jsonify({"error": "id is required"}), 400

    if from_month is None:
        return jsonify({"error": "from_month is required"}), 400

    # -----------------------------------------------------
    # 1️⃣ FIND ALL DEPOTS UNDER THIS DISTILLERY
    # -----------------------------------------------------
    distillery_csv = pd.read_csv("data/poc_distillery.csv")
    distillery_csv["from_entity_code"] = distillery_csv["from_entity_code"].astype(str)
    distillery_csv["to_entity_code"] = distillery_csv["to_entity_code"].astype(str)

    depots = distillery_csv[
        distillery_csv["from_entity_code"] == distillery_id
    ]["to_entity_code"].unique()

    if len(depots) == 0:
        return jsonify({"error": "No depots found under this distillery"}), 404

    # -----------------------------------------------------
    # 2️⃣ FIND RETAIL SHOPS UNDER THOSE DEPOTS
    # -----------------------------------------------------
    retail_shops = wholesale[
        wholesale["from_entity_code"].isin(depots)
    ]["to_entity_code"].unique()

    # -----------------------------------------------------
    # 3️⃣ SALES OF THOSE RETAIL SHOPS
    # -----------------------------------------------------
    dist_sales_full = retail[
        retail["entity_code"].isin(retail_shops)
    ]

    if dist_sales_full.empty:
        return jsonify({"error": "No retail sales found for this distillery"}), 404

    year = dist_sales_full["bill_date"].dt.year.max()

    # -----------------------------------------------------
    # 4️⃣ DEFINE TRAINING WINDOW
    # -----------------------------------------------------
    train_start_month = from_month - months
    train_end_month = from_month - 1

    train_start = datetime(year, train_start_month, 1)
    train_end = datetime(year, train_end_month, 1) + pd.offsets.MonthEnd(1)

    dist_sales_recent = dist_sales_full[
        (dist_sales_full["bill_date"] >= train_start) &
        (dist_sales_full["bill_date"] <= train_end)
    ]

    # ALL SKUs produced by this distillery (via depots)
    all_skus = dist_sales_full[["brand_name", "package_size"]].drop_duplicates().values.tolist()

    results = []

    # -----------------------------------------------------
    # 5️⃣ PROCESS EACH SKU
    # -----------------------------------------------------
    for brand, size in all_skus:

        sku_recent = dist_sales_recent[
            (dist_sales_recent["brand_name"] == brand) &
            (dist_sales_recent["package_size"] == size)
        ]

        sku_df = (
            sku_recent.groupby("bill_date")["sold_qty"]
            .sum().reset_index()
            .rename(columns={"bill_date": "ds", "sold_qty": "y"})
        )

        # Forecast demand
        if len(sku_df) >= 5:
            model = Prophet()
            model.fit(sku_df)

            total_demand = 0
            predict_year = year
            predict_month = from_month
        
            for _ in range(2):   # 2 months
                start = datetime(predict_year, predict_month, 1)
                end = start + pd.offsets.MonthEnd(1)
        
                future_dates = pd.date_range(start, end)
                forecast = model.predict(pd.DataFrame({"ds": future_dates}))
        
                # Add this month's predicted total
                total_demand += float(forecast["yhat"].sum())
        
                # Move to next month
                predict_month += 1
                if predict_month > 12:
                    predict_month = 1
                    predict_year += 1

            demand = total_demand
        else:
            demand = 0.0
        
        # -----------------------------------------------------
        # STOCK CALCULATIONS
        # -----------------------------------------------------

        # Distillery stock
        remaining_at_distillery = int(stock[
            (stock["entity_code"] == distillery_id) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum())

        # Depot stock (all depots under distillery)
        remaining_at_depot = int(stock[
            (stock["entity_code"].isin(depots)) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum())

        # Retail stock under those depots
        remaining_at_retail = int(stock[
            (stock["entity_code"].isin(retail_shops)) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum())

        remaining_stock = (
            remaining_at_distillery +
            remaining_at_depot +
            remaining_at_retail
        )

        quantitytoraise = int(round(max(demand - remaining_stock, 0)))

        results.append({
            "brand": str(brand),
            "package_size": str(size),
            "demand": int(round(demand)),
            "remaining_at_distillery": remaining_at_distillery,
            "remaining_at_depot": remaining_at_depot,
            "remaining_at_retail": remaining_at_retail,
            "remaining_stock": remaining_stock,
            "quantityToManufacture": quantitytoraise
        })

    return jsonify(results)


@app.route("/prophet/intent", methods=["POST"])
def predict_intent():
    req = request.json
    months = req.get("month", 2)
    from_month = req.get("from_month")
    distillery_id = str(req.get("id"))

    if distillery_id is None:
        return jsonify({"error": "id is required"}), 400

    if from_month is None:
        return jsonify({"error": "from_month is required"}), 400

    # -----------------------------------------------------
    # 1️⃣ FIND ALL DEPOTS UNDER THIS DISTILLERY
    # -----------------------------------------------------
    distillery_csv = pd.read_csv("data/poc_distillery.csv")
    distillery_csv["from_entity_code"] = distillery_csv["from_entity_code"].astype(str)
    distillery_csv["to_entity_code"] = distillery_csv["to_entity_code"].astype(str)

    depots = distillery_csv[
        distillery_csv["from_entity_code"] == distillery_id
    ]["to_entity_code"].unique()

    if len(depots) == 0:
        return jsonify({"error": "No depots found under this distillery"}), 404

    # -----------------------------------------------------
    # 2️⃣ FIND RETAIL SHOPS UNDER THOSE DEPOTS
    # -----------------------------------------------------
    retail_shops = wholesale[
        wholesale["from_entity_code"].isin(depots)
    ]["to_entity_code"].unique()

    # -----------------------------------------------------
    # 3️⃣ SALES OF THOSE RETAIL SHOPS
    # -----------------------------------------------------
    dist_sales_full = retail[
        retail["entity_code"].isin(retail_shops)
    ]

    if dist_sales_full.empty:
        return jsonify({"error": "No retail sales found for this distillery"}), 404

    year = dist_sales_full["bill_date"].dt.year.max()

    # -----------------------------------------------------
    # 4️⃣ DEFINE TRAINING WINDOW
    # -----------------------------------------------------
    train_start_month = from_month - months
    train_end_month = from_month - 1

    train_start = datetime(year, train_start_month, 1)
    train_end = datetime(year, train_end_month, 1) + pd.offsets.MonthEnd(1)

    dist_sales_recent = dist_sales_full[
        (dist_sales_full["bill_date"] >= train_start) &
        (dist_sales_full["bill_date"] <= train_end)
    ]

    # ALL SKUs produced by this distillery (via depots)
    all_skus = dist_sales_full[["brand_name", "package_size"]].drop_duplicates().values.tolist()

    results = []

    # -----------------------------------------------------
    # 5️⃣ PROCESS EACH SKU
    # -----------------------------------------------------
    for brand, size in all_skus:

        sku_recent = dist_sales_recent[
            (dist_sales_recent["brand_name"] == brand) &
            (dist_sales_recent["package_size"] == size)
        ]

        sku_df = (
            sku_recent.groupby("bill_date")["sold_qty"]
            .sum().reset_index()
            .rename(columns={"bill_date": "ds", "sold_qty": "y"})
        )

        # Forecast demand
        if len(sku_df) >= 5:
            model = Prophet()
            model.fit(sku_df)

            predict_start = datetime(year, from_month, 1)
            predict_end = predict_start + pd.offsets.MonthEnd(1)

            future_dates = pd.date_range(predict_start, predict_end)
            forecast = model.predict(pd.DataFrame({"ds": future_dates}))

            demand = float(forecast["yhat"].sum())
        else:
            demand = 0.0

        # -----------------------------------------------------
        # STOCK CALCULATIONS
        # -----------------------------------------------------

        # Distillery stock
        remaining_at_distillery = int(stock[
            (stock["entity_code"] == distillery_id) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum())

        # Depot stock (all depots under distillery)
        remaining_at_depot = int(stock[
            (stock["entity_code"].isin(depots)) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum())

        # Retail stock under those depots
        remaining_at_retail = int(stock[
            (stock["entity_code"].isin(retail_shops)) &
            (stock["brand_name"] == brand) &
            (stock["package_size"] == size)
        ]["closed_qty"].sum())

        remaining_stock = (
            remaining_at_distillery +
            remaining_at_depot +
            remaining_at_retail
        )

        quantitytoraise = int(round(max(demand - remaining_stock, 0)))

        results.append({
            "brand": str(brand),
            "package_size": str(size),
            "demand": int(round(demand)),
            "remaining_at_distillery": remaining_at_distillery,
            "remaining_at_depot": remaining_at_depot,
            "remaining_at_retail": remaining_at_retail,
            "remaining_stock": remaining_stock,
            "quantitytoraise": quantitytoraise
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
