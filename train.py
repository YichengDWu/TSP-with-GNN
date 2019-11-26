import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model_utils import loss_edges, load_dataset_for_regression
from models.model import TSPModel
from sklearn.utils.class_weight import compute_class_weight

def regress(model, bg, args):
        x = bg.ndata['coord']
        e = bg.edata['e']
        x, e = x.to(args['device']), e.to(args['device'])
        return model(bg, x, e)

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    for batch_id, batch_data in enumerate(data_loader):
        bg, ws, tours = batch_data
        ws = ws.to(args['device'])
        prediction = regress(model, bg, args)
        loss = loss_criterion(prediction, ws.long(), None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #train_meter.update(prediction, labels, masks)
        del prediction,bg,ws,tours
    print('Train loss at epoch {:d}/{:d}: {:.4f}'.format(
                    epoch+1,args['num_epochs'], loss.detach().item()))
    #torch.cuda.empty_cache()

def run_an_eval_epoch(args,epoch, model, data_loader,loss_criterion):
    model.eval()
    #eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            bg, ws, tours = batch_data
            ws = ws.to(args['device'])
            prediction = regress(model, bg,  args)
            loss = loss_criterion(prediction, ws.long(), None)
            del prediction,bg,ws,tours
            #eval_meter.update(prediction, labels, masks)
        #total_score = np.mean(eval_meter.compute_metric(args['metric_name']))   
        print('Val loss at epoch {:d}/{:d}: {:.4f}'.format(
                    epoch+1,args['num_epochs'], loss.detach().item()))
    torch.cuda.empty_cache()
    #return total_score

def main(args):
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = load_dataset_for_regression(args)
    
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=train_set.collate_tspgraphs)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=train_set.collate_tspgraphs)
    
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=train_set.collate_tspgraphs)

    if args['pre_trained']:
        args['num_epochs'] = 0
    else:
        model = TSPModel(args)
        loss_fn = loss_edges
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['decay_rate'])
        print(model)
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

        # Validation and early stop
        if (epoch+1) %5 == 0:
            val_score = run_an_eval_epoch(args,epoch, model, val_loader,loss_fn)
        #early_stop = stopper.step(val_score, model)
        #print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
        #    epoch + 1, args['num_epochs'], args['metric_name'], val_score,
         #   args['metric_name'], stopper.best_score))

        #if early_stop:
        #    break

    #if test_set is not None:
    #    if not args['pre_trained']:
     #       stopper.load_checkpoint(model)
     #   test_score = run_an_eval_epoch(args, model, test_loader)
     #   print('test {} {:.4f}'.format(args['metric_name'], test_score))

if __name__ == "__main__":
    import argparse

    from config import get_config
    parser = argparse.ArgumentParser(description='gcn_tsp_parser')
    parser.add_argument('-c','--config', type=str, default="configs/default.json")
    args = parser.parse_args()
    config_path = args.config
    config = get_config(config_path)
    main(config)