from model import *
from sampler import *
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
print(torch.__version__)

import numpy as np
import time

batch_size = 64
num_points = 64
num_labels = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)


def main(mixed_precision_enabled=False,
         scaler=None,
         static_compilation_enabled=False):

    pointnet = PointNet(num_points, num_labels)

    new_param = pointnet.state_dict()
    new_param['main.0.main.6.bias'] = torch.eye(3, 3).view(-1)
    new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
    pointnet.load_state_dict(new_param)

    optimizer = optim.Adam(pointnet.parameters(), lr=0.001)

    loss_list = []
    accuracy_list = []

    if static_compilation_enabled:
        pointnet = torch.compile(pointnet)

    pointnet.to(DEVICE)

    start_time = time.time_ns()

    if mixed_precision_enabled:
        print("Mixed Precision")

        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.functional.binary_cross_entropy_with_logits

        for iteration in range(1000 + 1):

            pointnet.zero_grad()

            input_data, labels = data_sampler(batch_size, num_points)
            input_data = input_data.cuda(device=DEVICE, non_blocking=True)
            labels = labels.cuda(device=DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda'):
                output = pointnet(input_data)
                # output = nn.Sigmoid()(output)

                error = criterion(output, labels)

            scaler.scale(error).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                output[output > 0.5] = 1
                output[output < 0.5] = 0
                accuracy = (output == labels).sum().item() / batch_size

            loss_list.append(error.item())
            accuracy_list.append(accuracy)

            if iteration % 10 == 0:
                print('Iteration : {}   Loss : {}'.format(
                    iteration, error.item()))
                print('Iteration : {}   Accuracy : {}'.format(
                    iteration, accuracy))
    else:
        print("Full Precision")

        criterion = nn.BCELoss()

        for iteration in range(1000 + 1):

            pointnet.zero_grad()

            input_data, labels = data_sampler(batch_size, num_points)
            input_data = input_data.cuda(device=DEVICE, non_blocking=True)
            labels = labels.cuda(device=DEVICE, non_blocking=True)

            output = pointnet(input_data)
            output = nn.Sigmoid()(output)

            error = criterion(output, labels)
            error.backward()

            optimizer.step()

            with torch.no_grad():
                output[output > 0.5] = 1
                output[output < 0.5] = 0
                accuracy = (output == labels).sum().item() / batch_size

            loss_list.append(error.item())
            accuracy_list.append(accuracy)

            if iteration % 10 == 0:
                print('Iteration : {}   Loss : {}'.format(
                    iteration, error.item()))
                print('Iteration : {}   Accuracy : {}'.format(
                    iteration, accuracy))

    end_time = time.time_ns()

    return np.min(loss_list), np.max(accuracy_list), end_time - start_time


if __name__ == '__main__':

    scaler = torch.amp.GradScaler('cuda')
    main()

    loss_1, acc_1, mt = main(
        mixed_precision_enabled=True,
        scaler=scaler,
    )

    # loss_1, acc_1, mt = main(
    #     mixed_precision_enabled=False,
    #     # scaler=scaler,
    #     static_compilation_enabled=True
    # )

    loss_2, acc_2, ft = main()

    print(
        f"Mixed Precision Time: {mt/10e9} seconds\nLoss: {loss_1}\tAcc: {acc_1}\n"
    )

    print(
        f"Full Precision Time: {ft/10e9} seconds\nLoss: {loss_2}\tAcc: {acc_2}\n"
    )

    print(f"Full - Mixed Difference: {(ft - mt)/10e9}")
    print(f"(Full - Mixed Difference)/Full: {((ft - mt)/ft)*100.0} %")
