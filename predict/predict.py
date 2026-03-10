import argparse
import pandas as pd
import json
from gears import PertData,GEARS
import os
import numpy as np

def to_jsonable(obj):
    """
    convert some data which is ndarray to jsonable and remove some data
    """
    import numpy as np

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "ts":
                continue
            out[k] = to_jsonable(v)
        return out

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    return obj

def load_data(data_path='./data',data_name='norman'):
    """
    load the data for building a gears model

    Parameters
    ----------
    data_path: str
    data_name: str

    returns
    -------
    pert_data: PertData
    """
    os.makedirs(data_path,exist_ok=True)
    pert_data=PertData(data_path)
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split='simulation', seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    return pert_data

def return_pert_names_and_condition(pert_data,save_path='./results'):
    """
    used to make a csv for users to read which perturbations can be predicted or conditions can be polt
    Parameters
    ----------
    pert_data: PertData
    save_path: str

    returns
    -------
    pert_names.csv
    available_conditions.csv
    """
    pert_names=pert_data.pert_names
    os.makedirs(save_path,exist_ok=True)
    df=pd.DataFrame(pert_names)
    df.to_csv(os.path.join(save_path,"pert_names.csv"),index=False,header=False)
    print("first five perturbations that can be predicted:")
    print(df.head())
    print(f"pert_names.csv is saved in {save_path}")
    conds = sorted(pert_data.adata.obs["condition"].unique())
    pd.DataFrame({"condition": conds}).to_csv(f"{save_path}/available_conditions.csv", index=False)

    return pert_names

def load_gears_model(pert_data,model_path,device="cuda"):
    """
    load the gears model for prediction

    Parameters
    ----------
    pert_data: PertData
    model_path: str
    device: str
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No such model file: {model_path}")
    model=GEARS(pert_data,device=device)
    model.load_pretrained(model_path)

    return model

def save_results(results,file_name,save_path='./results'):
    """
    save the prediction results as JSON,if JSON is existed, it will continue to write on JSON.
    Parameters
    ----------
    results: dict
    file_name: str
    save_path: str

    returns
    -------
    JSON file
    """
    os.makedirs(save_path,exist_ok=True)
    save_file=os.path.join(save_path,f"{file_name}.json")
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        old_data.update(results)
        data_to_save = old_data
    else:
        data_to_save = results
    with open(save_file,"w",encoding="utf-8") as f:
        json.dump(to_jsonable(data_to_save),f,indent=2)
    print(f"{file_name} saved in {save_path}")
    return

def run_prediction(model,
                   pert_list=None,
                   combo=None,
                   query=None,
                   GI_genes_file=None,
                   save_path="./pred_results",
                   save_file=None
                   ):
    """
    run prediction three mode:
    1.prediction:
        input(pert_list): FEV or FEV+AHR --> [['FEV']] or [['FEV'],[AHR]]
        output : predict_result.json or predict_result_with_uncertainty.json (decided by uncertainty or not)
    2.GI_predict:
        input(combo):FEV+AHR --> ['FEV','AHR']
        output: GI_predict_result.json
    3.plot_perturbation
        input(query): FOSB+CEBPB
        output: save_path/predict_{query}.png

    parameters:
    ----------
    pert_list: list
    combo: str
    query: str
    GI_genes_file: str
    save_path: str
    save_file: str (path used for save png)

    returns
    -------
    prediction: predict_result.json or predict_result_with_uncertainty.json
    GI_predict: GI_predict_result.json
    plot_perturbation: predict_{query}.png
    """
    def str_to_gene_list(s):
        return [[x.strip() for x in s.split("+")]]
    def str_to_combo(s):
        return [x.strip() for x in s.split("+")]
    if pert_list is not None:
        pert = str_to_gene_list(pert_list)
        result = model.predict(pert_list=pert)
        if isinstance(result, tuple) and len(result) == 2:
            file_name="predict_result_with_uncertainty"
            results_pred, results_logvar_sum = result
            out = {f"{pert_list}":
                {
                "predict_result": results_pred,
                "predict_uncertainty": results_logvar_sum
                }
            }

            save_results(out, file_name, save_path=save_path)
            print(f"predict {pert_list} successfully with uncertainty")
        else:
            file_name="predict_result"
            out = {f"{pert_list}":
                {
                    "predict_result": result
                }
            }
            save_results(out,file_name,save_path=save_path)
            print(f"predict {pert_list} successfully")
    if combo is not None:
        combo_str = str_to_combo(combo)
        result = model.GI_predict(combo=combo_str,GI_genes_file=GI_genes_file)
        out = {f"{combo}":
                   {
                    "GI_predict": result
                   }
               }
        file_name="GI_predict_result"
        save_results(out,file_name,save_path=save_path)

    if query is not None:
        if save_path is not None:
            save_file=os.path.join(save_path,f"predict_{query}.png")
            model.plot_perturbation(query, save_file=save_file)
        else:
            raise ValueError("save_file is None")
    return


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",default="./data")
    ap.add_argument("--data_name",default="norman")
    ap.add_argument("--save_path",default="./pred_results")
    ap.add_argument("--pert_list",default=None,type=str)
    ap.add_argument("--combo",default=None,type=str)
    ap.add_argument("--query",default=None,type=str)
    ap.add_argument("--model_path",default=None,type=str)
    ap.add_argument("--save_file",default=None,type=str)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--export",action="store_true",default=False)
    return ap.parse_args()

def main():
    args = parse_args()
    pert_data=load_data(args.data_path,args.data_name)
    if args.export:
        return_pert_names_and_condition(pert_data,save_path=args.save_path)
        return
    gears_model=load_gears_model(pert_data,model_path=args.model_path,device=args.device)

    run_prediction(gears_model,
                   pert_list=args.pert_list,
                   combo=args.combo,
                   query=args.query,
                   save_path=args.save_path,
                   save_file=args.save_file
                   )
if __name__ =="__main__":
    main()












