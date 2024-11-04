# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 03:21:35 2024

@author: LG
"""

import math  # math.ceil을 사용할 경우 필요한 라이브러리
import pandas as pd  # pandas 라이브러리 추가
from ortools.linear_solver import pywraplp
import json

# Load data from JSON file
with open(r'C:\smartfarm\smartfarm_data\scenario_1.json') as f:
    data = json.load(f)

# Extract data from JSON
stores = data["stores"]
farms = data["farms"]
crops = data["crops"]

D = {tuple(map(int, key.strip("()").split(","))): value for key, value in data["demand"].items()}
P = {int(key): value for key, value in data["production_rate"].items()}
CS = {int(key): value for key, value in data["store_costs"].items()}
CF = {int(key): value for key, value in data["farm_costs"].items()}
PC = {int(key): value for key, value in data["penalty_cost"].items()}
B = {int(key): value for key, value in data["selling_price"].items()}
MCS = {int(key): value for key, value in data["store_capacity"].items()}
MCF = {int(key): value for key, value in data["farm_capacity"].items()}

Q = data["transport_unit"]  # 운송 단위
DC = data["transport_cost"]  # 운송 비용

DF = {tuple(map(int, key.strip("()").split(","))): value for key, value in data["farm_store_distance"].items()}
DS = {tuple(map(int, key.strip("()").split(","))): value for key, value in data["store_store_distance"].items()}

# 모델 초기화
solver = pywraplp.Solver.CreateSolver('SCIP')

# G[k] 정의 - 항상 올림
G = {(i, k): math.ceil(D[i, k] / P[k]) for i in stores for k in crops}  # 올림된 G[k] 계산

# 결정변수 정의
x = {(i, k): solver.IntVar(0, solver.infinity(), f'x_{i}_{k}') for i in stores for k in crops}
z = {(j, i, k): solver.IntVar(0, solver.infinity(), f'z_{j}_{i}_{k}') for j in farms for i in stores for k in crops}
w = {(s, i, k): solver.IntVar(0, solver.infinity(), f'w_{s}_{i}_{k}') for i in stores for s in stores if i != s for k in crops}
l = {(i, k): solver.IntVar(0, solver.infinity(), f'l_{i}_{k}') for i in stores for k in crops}

# 추가 이진 변수 (운송 비용의 올림 처리를 위한 변수)
ceil_z = {(j, i): solver.IntVar(0, solver.infinity(), f'ceil_z_{j}_{i}') for j in farms for i in stores}
ceil_w = {(s, i): solver.IntVar(0, solver.infinity(), f'ceil_w_{s}_{i}') for i in stores for s in stores if i != s}

# 목적 함수 정의
objective_terms = []
for k in crops:
    # Objective term for total sales
    total_sales = B[k] * (
        solver.Sum(x[i, k] for i in stores) +
        solver.Sum(w[s, i, k] for s in stores for i in stores if s != i) +
        solver.Sum(z[j, i, k] for j in farms for i in stores)
    )
    objective_terms.append(total_sales)

    # Store costs
    store_costs = -CS[k] * solver.Sum(x[i, k] for i in stores)
    objective_terms.append(store_costs)

    # Store costs from other stores
    store_costs_from_others = -CS[k] * solver.Sum(w[s, i, k] for s in stores for i in stores if i != s)
    objective_terms.append(store_costs_from_others)

    # Farm costs
    farm_costs = -CF[k] * solver.Sum(z[j, i, k] for j in farms for i in stores)
    objective_terms.append(farm_costs)

# 전체 운송 비용 계산 (작물별로 합치지 않음)
transport_costs_stores = -solver.Sum(ceil_w[s, i] * DC * DS[s, i] for i in stores for s in stores if s != i)
objective_terms.append(transport_costs_stores)

transport_costs_farms = -solver.Sum(ceil_z[j, i] * DC * DF[j, i] for j in farms for i in stores)
objective_terms.append(transport_costs_farms)

# Penalty costs
for k in crops:
    penalty_costs = -PC[k] * solver.Sum(l[i, k] for i in stores)
    objective_terms.append(penalty_costs)

# Objective function maximization
solver.Maximize(solver.Sum(objective_terms))

# 수요 만족 제약식
for i in stores:
    for k in crops:
        solver.Add(x[i, k] + solver.Sum(w[s, i, k] for s in stores if s != i) + solver.Sum(z[j, i, k] for j in farms) <= G[i, k])

# 농장 용량 제한
for j in farms:
    solver.Add(solver.Sum(z[j, i, k] for k in crops for i in stores) <= MCF[j])

# 스토어 용량 제한
for i in stores:
    solver.Add(solver.Sum(x[i, k] for k in crops) + solver.Sum(w[i, s, k] for s in stores if i != s for k in crops) <= MCS[i])

# 부족량 정의 제약식
for i in stores:
    for k in crops:
        solver.Add(l[i, k] >= D[i, k] / P[k] - (x[i, k] + solver.Sum(w[s, i, k] for s in stores if s != i) + solver.Sum(z[j, i, k] for j in farms)))

# 올림 연산 제약 (스토어 간 운송량에 대한 올림 처리)
for i in stores:
    for s in stores:
        if i != s:
            for k in crops:
                total_w = solver.Sum(w[s, i, k] for k in crops)
                # 올림 조건을 제약식으로 구현
                solver.Add(ceil_w[s, i] >= total_w / Q)
                solver.Add(ceil_w[s, i] <= (total_w + Q - 1) / Q)

# 올림 연산 제약 (농장에서 스토어로의 운송량에 대한 올림 처리)
for j in farms:
    for i in stores:
        for k in crops:
            total_z = solver.Sum(z[j, i, k] for k in crops)
            # 올림 조건을 제약식으로 구현
            solver.Add(ceil_z[j, i] >= total_z / Q)
            solver.Add(ceil_z[j, i] <= (total_z + Q - 1) / Q)

# 해결
status = solver.Solve()

# 해결 결과 출력
if status == pywraplp.Solver.OPTIMAL:
    print(f"목적함수: {round(solver.Objective().Value())}")

    # 목적 함수 구성 요소 출력
    print("\n===== 목적 함수 구성 요소 =====")
    for k in crops:
        # 계산된 값
        sales = B[k] * (
            sum(x[i, k].solution_value() for i in stores) +
            sum(w[s, i, k].solution_value() for s in stores for i in stores if s != i) +
            sum(z[j, i, k].solution_value() for j in farms for i in stores)
        )
        store_cost = -CS[k] * sum(x[i, k].solution_value() for i in stores)
        store_cost_other = -CS[k] * sum(w[s, i, k].solution_value() for s in stores for i in stores if i != s)
        farm_cost = -CF[k] * sum(z[j, i, k].solution_value() for j in farms for i in stores)
        transport_cost_stores = -sum(ceil_w[s, i].solution_value() * DC * DS[s, i] for i in stores for s in stores if s != i)
        transport_cost_farms = -sum(ceil_z[j, i].solution_value() * DC * DF[j, i] for j in farms for i in stores)
        penalty_cost = -PC[k] * sum(l[i, k].solution_value() for i in stores)

        total_objective = sales + store_cost + store_cost_other + farm_cost + transport_cost_stores + transport_cost_farms + penalty_cost

        print(f"작물 {k}: 판매수익 = {sales}, 스토어 비용 = {store_cost}, 다른 스토어 비용 = {store_cost_other}, 농장 비용 = {farm_cost}, "
              f"스토어 운송 비용 = {transport_cost_stores}, 농장 운송 비용 = {transport_cost_farms}, 패널티 비용 = {penalty_cost}, "
              f"총합 = {total_objective}")

else:
    print(f"해를 찾지 못했습니다.")

# 해결
status = solver.Solve()

# 해결 결과 출력
if status == pywraplp.Solver.OPTIMAL:
    print(f"목적함수: {round(solver.Objective().Value())}")

    print("\n===== 스토어별 물량 수령 현황 =====")
    for i in stores:
        print(f"\n### 스토어 {i} ###")
        for k in crops:
            # 자가 생산량 (정수로 반올림)
            produced = round(x[i, k].solution_value())
            # 농장에서 받은 양 (정수로 반올림)
            received_from_farms = round(sum(z[j, i, k].solution_value() for j in farms))
            # 다른 스토어에서 받은 양 (정수로 반올림)
            received_from_stores = round(sum(w[s, i, k].solution_value() for s in stores if s != i))

            # 총합 (자가 생산 + 농장에서 받은 양 + 다른 스토어에서 받은 양)
            total_received = produced + received_from_farms + received_from_stores

            # 포기하는 수량 계산 (l 변수 사용)
            discarded = round(l[i, k].solution_value())  # l 변수를 사용하여 포기량 계산

            print(f"작물 {k}: 총합 = {total_received} (자가 생산 = {produced}, 농장에서 받은 양 = {received_from_farms}, 다른 스토어에서 받은 양 = {received_from_stores}, 포기량 = {discarded})")

        # 농장에서 스토어로 받은 물량 세부 사항 출력
        print(f"\n### 농장으로부터 받은 물량 ###")
        for j in farms:
            for k in crops:
                received = round(z[j, i, k].solution_value())
                print(f"  농장 {j}에서 받은 작물 {k}의 양: {received}")

        # 다른 스토어에서 받은 물량 세부 사항 출력
        print(f"\n### 다른 스토어로부터 받은 물량 ###")
        for s in stores:
            if s != i:
                for k in crops:
                    received_from_store = round(w[s, i, k].solution_value())
                    print(f"  스토어 {s}에서 받은 작물 {k}의 양: {received_from_store}")

    # 결정변수 값 출력
    print("\n===== 결정 변수 값 =====")
    for i in stores:
        for k in crops:
            x_value = round(x[i, k].solution_value())
            print(f"X{i}{k}: {x_value}")

    for i in stores:
        for s in stores:
            if i != s:
                for k in crops:
                    w_value = round(w[i, s, k].solution_value())
                    print(f"W{s}{i}{k}: {w_value}")

    for j in farms:
        for i in stores:
            for k in crops:
                z_value = round(z[j, i, k].solution_value())
                print(f"Z{j}{i}{k}: {z_value}")

    for i in stores:
        for k in crops:
            l_value = round(l[i, k].solution_value())
            print(f"L{i}{k}: {l_value}")

# 해결
status = solver.Solve()

# 해결 결과 출력
if status == pywraplp.Solver.OPTIMAL:
    results = []

    # 결과 수집
    for i in stores:
        for k in crops:
            produced = round(x[i, k].solution_value())
            received_from_farms = round(sum(z[j, i, k].solution_value() for j in farms))
            received_from_stores = round(sum(w[s, i, k].solution_value() for s in stores if s != i))
            total_received = produced + received_from_farms + received_from_stores
            discarded = round(l[i, k].solution_value())

            # Collect farms information
            farms_info = []
            for j in farms:
                if round(z[j, i, k].solution_value()) > 0:  # If there's a contribution from this farm
                    farms_info.append(j)

            # Collect stores information
            stores_info = []
            for s in stores:
                if s != i and round(w[s, i, k].solution_value()) > 0:  # Exclude self
                    stores_info.append(s)

            results.append({
                "Store": i,
                "Crop": k,
                "Produced": produced,
                "Received from Farms": received_from_farms,
                "Farms Received From": ", ".join(map(str, farms_info)),  # Join farm IDs as a string
                "Received from Stores": received_from_stores,
                "Stores Received From": ", ".join(map(str, stores_info)),   # Join store IDs as a string
                "Total": total_received,
                "Discarded": discarded,
            })

    # 결과를 데이터프레임으로 변환
    df = pd.DataFrame(results)

    # CSV 파일로 저장
    output_path = r"C:\smartfarm\output\results_1.csv"
    df.to_csv(output_path, index=False)

    print(f"결과가 {output_path}에 저장되었습니다.")
else:
    print(f"해를 찾지 못했습니다.")
