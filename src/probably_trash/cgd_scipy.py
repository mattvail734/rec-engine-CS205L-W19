            for u, i, r in trainset.all_ratings():
                args = [u, i, r, global_mean, reg_bu, bu, reg_bi, bi, reg_pu, pu, reg_qi, qi]
                print('pu one user row outside objective {}'.format(args[9][args[0], :]))
                print('qi one movie row outside objective {}'.format(args[11][args[1], :]))
                def objective_bu(bu, args):
                    bui = args[3] + bu[args[0]] + args[7][args[1]]
                    puqi = np.dot(args[9][args[0], :], args[11][args[1], :])
                    residual = bui + puqi
                    bui_reg = args[4] * bu[args[0]] + args[6] * args[7][args[1]]
                    puqi_reg = args[8] * norm(args[9][args[0], :]) + args[10] * norm(args[11][args[1], :])
                    regularization = bui_reg + puqi_reg
                    return residual + regularization

                def objective_bi(bi, args):
                    bui = args[3] + args[5][args[0]] + bi[args[1]]
                    puqi = np.dot(args[9][args[0], :], args[11][args[1], :])
                    residual = bui + puqi
                    bui_reg = args[4] * args[5][args[0]] + args[6] * bi[args[1]]
                    puqi_reg = args[8] * norm(args[9][args[0], :]) + args[10] * norm(args[11][args[1], :])
                    regularization = bui_reg + puqi_reg
                    return residual + regularization

                def objective_pu(pu, args):
                    print('pu inside objective: {}'.format(pu))
                    print('qi inside objective: {}'.format(pu))
                    bui = args[3] + args[5][args[0]] + bi[args[1]]
                    puqi = np.dot(pu[args[0], :], args[11][args[1], :])
                    residual = bui + puqi
                    bui_reg = args[4] * args[5][args[0]] + args[6] * bi[args[1]]
                    puqi_reg = args[8] * norm(pu[args[0], :]) + args[10] * norm(args[11][args[1], :])
                    regularization = bui_reg + puqi_reg
                    return residual + regularization

                def objective_qi(qi, args):
                    bui = args[3] + args[5][args[0]] + bi[args[1]]
                    puqi = np.dot(args[9][args[0], :], qi[args[1], :])
                    residual = bui + puqi
                    bui_reg = args[4] * args[5][args[0]] + args[6] * bi[args[1]]
                    puqi_reg = args[8] * norm(args[9][args[0], :]) + args[10] * norm(qi[args[1], :])
                    regularization = bui_reg + puqi_reg
                    return residual + regularization

                options = {'maxiter': 3}
                # update bu using CG
                bu = minimize(objective_bu, bu, args, method='CG', options=options).x
                print('bu after cg: {}'.format(bu))
                # update bi using CG
                bi = minimize(objective_bi, bi, args, method='CG', options=options).x
                # update pu using CG
                pu = minimize(objective_pu, pu, args, method='CG', options=options).x
                # update qi using CG
                qi = minimize(objective_qi, qi, args, method='CG', options=options).x

                def objective(bu, u, i, r, global_mean, reg_bu, reg_bi, bi, reg_pu, pu, reg_qi, qi):
                    bui = global_mean + bu[u] + bi[i]
                    puqi = np.dot(pu[u, :], qi[i, :])
                    residual = bui + puqi
                    bui_reg = reg_bu * bu[u] + reg_bi * bi[i]
                    puqi_reg = reg_pu * norm(pu[u, :]) + reg_qi * norm(qi[i, :])
                    regularization = bui_reg + puqi_reg
                    return residual + regularization
