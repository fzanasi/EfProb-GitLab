#
# Connections between python's Probabilistic Graphical Model (pgm)
# library pgmpy and EfProb, in particular with respect to Bayesian
# network models and inference
#
# Copyright: Bart Jacobs; 
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2017-12-27
#
from efprob_dc import *
from baynet import *

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#from pgmpy.inference import BeliefPropagation

import pydot
from PIL import Image


#####################################################################
#
# The main functions in this file are:
#
# - stretch: linearising a Bayesian model in pgmpy to a list of
#   composable EfProb channels, where the first entry is a state
#
# - inference_query: it computes, starting from the output of the
#   stretch function, the marginal distribution, given a list of
#   evidence.
#
#####################################################################


#
# Auxiliary functions, from pgm to efprob (or pydot)
#

#
# Turn a name (string) and cardinality (number) into a named EfProb
# domain, with entiries of the form name_i
#
def efprob_domain(name, card):
    items = [ name + "_" + str(i) for i in range(card)]
    return Dom(items, names = Name(name))
    
#
# Turn the nodes of a graphical model into a list of EfProb domains
#
def efprob_domains_of_pgm(model):
    nodes = model.nodes()
    domains = {}
    for n in nodes:
        domains[n] = efprob_domain(n, model.get_cardinality(n))
    return domains

#
# The incoming edges of a node in a graphical model are turned into a
# EfProb channel. This is done for each node, and yields a dictionary
# with a channel for each node (as key)
#
def efprob_channels_of_pgm(model):
    nodes = model.nodes()
    cpts = {}
    for n in nodes:
        cpd = model.get_cpds(node=n)
        card_n = cpd.variable_card
        # first extract codomain
        cod = efprob_domain(cpd.variable, card_n)
        reverse_parent_nodes = cpd.get_evidence()
        parent_nodes = [reverse_parent_nodes[i] 
                        for i in range(len(reverse_parent_nodes)-1, -1, -1)]
        parents_num = len(parent_nodes)
        #print("* Handling node:", n, card_n, parent_nodes )
        vals = cpd.get_values() 
        if parents_num == 0:
            # include a state, not a channel, since there is no domain
            cpts[n] = State(vals.transpose()[0], cod)
        else:
            # extract a channel; first find its domain
            doms = [efprob_domain(pn, model.get_cardinality(pn)) 
                    for pn in parent_nodes]
            dom = reduce(lambda d1, d2: d1 + d2, doms)
            #print("Domain:", len(dom), dom)
            states = []
            steps = 1
            for i in range(parents_num):
                parent_i = parent_nodes[i]
                card_i = model.get_cardinality(parent_i)
                steps *= card_i
            states = [State(vals.transpose()[j], cod) for j in range(steps)]
            cpts[n] = chan_from_states(states, dom)
    return cpts

#
# Extra a pydot graph of a graphical model
#
def pydot_graph_of_pgm(model):
    graph = pydot.Dot(graph_type='digraph')
    nodes = model.nodes()
    for n in nodes:
        graph.add_node(pydot.Node(n))
    edges = model.edges()
    for e in edges:
        graph.add_edge(pydot.Edge(e))
    return graph


#
# Turn a Probabilistic Graphical Model (pgm, as in pgmpy) into a
# sequence of channels, together with a node_pointer dictionary which
# tells for each node in the model where it occurs in this sequence.
#
# Optionally, via additional True/False settings
#
# - a pydot graph is produced of the pgm (with additional copy and
#   identity nodes), indicating the burocratic channels for copying
#   and reordering
#
# - an 'observed' version, both of the sequence of channels and of the
#   graph, is produced, in which internal nodes become 'observable'
#   (or accessible). This version is useful for producing the joint
#   distribution from the model.
#
def stretch(pgm, graph_output=True, observed=False):
    #
    # Extraction of relevant data from input
    #
    nodes = pgm.nodes
    efprob_domains = efprob_domains_of_pgm(pgm)
    channels = efprob_channels_of_pgm(pgm)
    parents = {}
    children = {}
    for n in pgm.nodes():
        parents[n] = []
        children[n] = []
    for e in pgm.edges():
        children[e[0]].append(e[1])
        parents[e[1]].append(e[0])
    #
    # Initialisation of data structures that will be build up: (a)
    # list of channels, (b) nodes pointers, (c) node copies, (d)
    # graph, optionally
    #
    # (a) List of channels, where the first entry consists of the
    # (product of the) initial states. This list of channels is the
    # main part of the output
    #
    channel_list = []
    # 
    # (b) Dictionary with index level (natural number) for each node
    #
    node_pointer = {}
    pointer = 2
    #
    # (c) Dictionary for keeping track of how many copies are needed of a
    # node, so that copying can be done lazily, whenever needed.
    #
    node_copies = {}
    #
    # (d) Optional graph of the linearised graph; original nodes are
    # in green
    #
    if graph_output:
        stretched_graph = pydot.Dot(graph_type='digraph')
    #
    # List of available nodes for finding a match of
    # channels. Elements of this list will be pairs, consisting of the
    # node (name) 'n' together with its name node_name(n,i) in the
    # graph. Since nodes may occur multiple times, we need to use
    # different names in the graph.
    #
    available_nodes = []
    #
    # Step 1: handling of initial nodes; 
    #
    # deep copy
    unprocessed_nodes = [n for n in nodes]
    initial_nodes = []
    for n in nodes:
        # print("Handling node: ", n)
        parents_n = parents[n]
        if len(parents_n) == 0:
            # n is an initial node
            initial_nodes.append(n)
            unprocessed_nodes.remove(n)
            node_pointer[n] = 0          
            if graph_output:
                stretched_graph.add_node(pydot.Node(n, 
                                                    style="filled", 
                                                    fillcolor="green"))
            if observed:
                available_nodes.append(n)
            children_n = children[n] 
            children_num = len(children_n)
            # print("Children of: ", n, children_n)
            #
            # Make copies of the initial states, depending on the
            # number of children (and on whether or not we want an
            # 'observed' version)
            #
            if children_num > 0:
                # n is not a final node
                available_nodes.append(n)
                node_copies[n] = children_num + (1 if observed else 0)
                if graph_output:
                    stretched_graph.add_node(pydot.Node(n + "!copy", 
                                                        width=0.15,
                                                        height=0.15,
                                                        fixedsize=True,
                                                        style="filled", 
                                                        fillcolor="black",
                                                        fontsize=0))
                    stretched_graph.add_edge(pydot.Edge(n, n + "!copy"))
                    if observed:
                        stretched_graph.add_edge(pydot.Edge(n + "!copy",
                                                            " " + n + " "))
    if len(initial_nodes) == 0:
        raise Exception('Error: the model does not have initial nodes')
    initial_state = reduce(lambda s1, s2: s1 @ s2, [channels[n]
                                            for n in initial_nodes])
    #
    # Put the initial state at the beginning of the channel list, and
    # add an identity channel. The latter may be turned into a copy
    # channel later on.
    #
    channel_list.append(initial_state)
    channel_list.append(idn(initial_state.dom))
    # print("Initial state: ", channel_list[1] >> initial_state )
    #
    # Step 2: continue with non-initial nodes from the model: cycle
    # through the set of unprocessed nodes until all associated
    # channels have been applied to the current state, and the list of
    # unprocessed nodes is empty.
    #
    iterations = 0
    while len(unprocessed_nodes) > 0:
        #if iterations > 10:
        #    break
        for un in unprocessed_nodes:
            iterations += 1
            un_chan = channels[un]
            parents_un = [n.name for n in un_chan.dom.names]
            num_parents_un = len(parents_un)
            current_dom = channel_list[len(channel_list)-1].cod
            dom_lens = [len(d) for d in current_dom]
            print("* Iteration", iterations, "for", un, "of size:", dom_lens,
                  "is", reduce(operator.mul, dom_lens, 1))
            #
            # Just check if all parents nodes are available (not yet where)
            #
            proceed = all([pn in available_nodes for pn in parents_un])
            if not proceed:
                # handle this node un later
                continue
            # print("Found parents ", parents_un, "in", available_nodes, proceed)
            # print("Copies", node_copies)
            #
            # Make list for the nodes that should be copied
            #
            copy_list = len(available_nodes) * [1]
            new_available_nodes = []
            for i in range(len(available_nodes)):
                n_i = available_nodes[i]
                new_available_nodes.append(n_i)
                if n_i in parents_un:
                    node_copies[n_i] -= 1
                    if node_copies[n_i] > 0:
                        # more copies of un needed later on
                        new_available_nodes.append(n_i)
                        copy_list[i] += 1
            available_nodes = new_available_nodes
            # update the last channel with the required copying
            lcs = len(channel_list)
            last_channel = channel_list[lcs-1]
            channel_list[lcs-1] = copy_chan(last_channel, copy_list)
            #
            # Now search for the precise positions of the parent nodes
            #
            # print("==> Searching locations of parents", parents_un, "with availables", available_nodes)
            search_copy_of_nodes = [u for u in available_nodes]
            swaps = list(range(len(available_nodes)))
            #
            # find the actual occurrences of un's parents in available domains
            #
            for i in range(num_parents_un):
                # print("... searching for parent: ", parents_un[i] )
                #
                # try to locate i-th parent among domains
                #
                for j in range(len(available_nodes)):
                    if search_copy_of_nodes[j] == parents_un[i]:
                        # print("Parent", i, "found at", j)
                        tmp = swaps[j]
                        swaps[j] = swaps[i]
                        swaps[i] = tmp
                        search_copy_of_nodes[j] = search_copy_of_nodes[i]
                        search_copy_of_nodes[i] = parents_un[i]
                        break
            print("==> All parents located of:", un)
            # print("Available domains: ", available_nodes)
            #
            # incorporate swaps into available nodes and arguments
            #
            inv_swaps = len(available_nodes) * [0]
            swapped_doms = []
            swapped_available_nodes = []
            for i in range(len(available_nodes)):
                inv_swaps[swaps[i]] = i
                swapped_available_nodes.append(available_nodes[swaps[i]])
            available_nodes = swapped_available_nodes
            # print(swaps, inv_swaps, available_nodes)
            swapped_doms = [efprob_domains[n] for n in swapped_available_nodes]
            swapped_dom = reduce(lambda d1, d2: d1 + d2, swapped_doms)
            # print(swapped_dom)
            #
            # Build the channel that does the swapping
            #
            if graph_output:
                stretched_graph.add_node(pydot.Node(un, 
                                                    style="filled", 
                                                    fillcolor="green"))
                for i in range(num_parents_un):
                    stretched_graph.add_edge(pydot.Edge(
                        available_nodes[i] + "!copy", un))
            un_chan_id = un_chan
            diff = len(available_nodes) - num_parents_un
            if diff > 0:
                identities_doms = []
                for i in range(diff):
                    d_i = swapped_dom.get_nameditem(i + num_parents_un)
                    identities_doms.append(d_i)
                identities_dom = reduce(lambda d1, d2: d1 + d2, 
                                        identities_doms)
                identities = idn(identities_dom)
                un_chan_id = un_chan @ identities
            #
            # Add the channel to the list, with its domains permuted
            # 
            channel_list.append(perm_chan(un_chan_id, 
                                          dom_perm = inv_swaps))
            pointer += 1
            node_pointer[un] = pointer
            # print("Available nodes", available_nodes)
            available_nodes = [un] + available_nodes[num_parents_un:]
            # print("Remaining nodes", available_nodes)
            #
            # Finally, update the graph
            #
            childred_un = children[un] 
            num_children_un = len(childred_un)
            node_copies[un] = num_children_un + (1 if observed else 0)
            #print("Children of: ", un, childred_un)
            if num_children_un > 0:
                if graph_output:
                    stretched_graph.add_node(pydot.Node(un + "!copy", 
                                                        width=0.15,
                                                        height=0.15,
                                                        fixedsize=True,
                                                        style="filled", 
                                                        fillcolor="black",
                                                        fontsize=0))
                    stretched_graph.add_edge(pydot.Edge(un, un + "!copy"))
                    if observed:
                        stretched_graph.add_edge(pydot.Edge(un + "!copy",
                                                            " " + un + " "))
            #
            # Terminate handling node un and return to for-loop
            #
            unprocessed_nodes.remove(un)
    #
    # Collect the results in a dictionary
    # 
    result = {
        'pointer'  : node_pointer,
        'channels' : channel_list
    }
    if graph_output:
        result['graph'] = stretched_graph
    return result


#
# Successively apply state transformation >> to the initial state, by
# applying the successive channels one-by-one. The chan-list input is
# typically produced by the stretch function.
#
def evaluate_stretch(chan_list):
    state = chan_list[0]
    for i in range(len(chan_list)-1):
        state = chan_list[i+1] >> state
    return state

#
# Return a list of (first) marginals from a channel-list and a pointer
# dictionary, as produced by the stretch function. 
#
def marginals_stretch(stretch_dict):
    chan_list = stretch_dict['channels']
    state_list = [chan_list[0]]
    for i in range(len(chan_list) - 1):
        state_list.append(chan_list[i+1] >> state_list[i])
    ptr = stretch_dict['pointer']
    result = []
    for n in ptr.keys():
        if ptr[n] > 0:
            # print(n, ptr[n], len(state_list) )
            state = state_list[ptr[n]-1]
            mask = len(state.dom) * [0]
            mask[0] = 1
            result.append(state % mask)
    return result


#
# This function operates on the output of the 'stretch' function and
# uses it for inference. It takes two additional arguments:
#
# - 'marginal', which is a node in the original model, at which
#   position the probability is computed, given the evidence
#
# - 'evidence_dict', which is a dictionary mapping nodes to list of
#   probabilities in [0,1], acting as predicate values. If the node
#   has cardinality n, the list must have length n.
#
def inference_query(stretch_dict, marginal, evidence_dict):
    chan_list = stretch_dict['channels']
    chan_list_len = len(chan_list)
    state = chan_list[0]
    init_pred = truth(state.dom)
    ptr = stretch_dict['pointer']
    ptrkeys = ptr.keys()
    evkeys = evidence_dict.keys()
    if not (marginal in ptrkeys):
        raise Exception('Marginal does not occur')
    if not any([k in ptrkeys for k in evkeys]):
        raise Exception('Some of the evidence keys do not occur')
    #
    # The evidence is inserted into a list in parallel to the list of
    # channels, so that updates can happen locally. This evidence-list
    # contains two kinds of entries: 
    #
    # - (0,None) --- nothing will happen here 
    # - (1,pred)
    #
    evidence_list = chan_list_len * [(0,None)]
    for k in evkeys:
        position_k = ptr[k]
        if position_k == 0:
            #
            # evidence at an initial node; the state is then updated
            # immediately with the corresponding (weakened) predicate;
            # nothing is added to the evidence list.
            #
            dom = state.dom
            preds = []
            for i in range(len(dom)):
                if k == dom.names[i].name:
                    preds.append(Predicate(evidence_dict[k], dom[i]))
                else:
                    preds.append(truth(dom[i]))
            weakened_pred = reduce(lambda p1, p2: p1 @ p2, preds)
            state = state / weakened_pred
        else:
            dom = chan_list[position_k-1].cod
            #print("Evidence at:", k, position_k, chan_list_len, dom)
            #
            # By construction, the domain of the predicate is at the
            # first position in the codomain of channel at this
            # position
            #
            pred = Predicate(evidence_dict[k], dom[0])
            if len(dom) > 0:
                # weaken
                pred = pred @ truth(dom[1:])
            evidence_list[position_k - 1] = (1,pred)
    position_marginal = ptr[marginal]
    #
    # Start with forward state transformation up to the marginal
    # position, updating with predicates, if any, along the way
    #
    for i in range(position_marginal-1):
        if evidence_list[i][0] == 0:
            # no predicate at this point, transform existing state
            state = chan_list[i+1] >> state
        else:
            # update the current state with the predicate first, then
            # transform, i.e. do forward inference
            state = state / evidence_list[i][1]
            state = chan_list[i+1] >> state
    #
    # Continue with backward predicate transformation, starting from
    # the truth predicate, adding predicates, if any, along the way,
    # via conjunction &
    #
    pred = truth(chan_list[chan_list_len-1].cod)
    for i in range(chan_list_len - position_marginal):
        if evidence_list[chan_list_len - i - 1][0] == 0:
            # no predicate at this point, transform existing predicate
            pred = chan_list[chan_list_len - i - 1] << pred
        else:
            # take the conjunction of the current predicate with the
            # predicate at this point, then transform
            pred = pred & evidence_list[chan_list_len - i - 1][1]
            pred = chan_list[chan_list_len - i - 1] << pred
    #
    # Combine the results of the forward and backward operations by
    # updating the state once more, and taking the appropriate (first)
    # marginal.
    #
    state = state / pred
    mask = len(state.dom) * [0]
    mask[0] = 1
    return state % mask




"""

Older, "eager copy" version

def stretch(pgm, graph_output=True, observed=False):
    #
    # Extraction of relevant data from input
    #
    nodes = pgm.nodes
    domains = efprob_domains_of_pgm(pgm)
    channels = efprob_channels_of_pgm(pgm)
    parents = {}
    children = {}
    for n in pgm.nodes():
        parents[n] = []
        children[n] = []
    for e in pgm.edges():
        children[e[0]].append(e[1])
        parents[e[1]].append(e[0])
    #
    # Initialisation of data structures that will be built up:
    # (a) list of channels, (b) nodes pointers, (c) graph, optionally
    #
    # (1) List of channels, where the first entry consists of the
    # (product of the) initial states.
    #
    channel_list = []
    # 
    # (b) Dictionary with index level (natural number) for each node
    #
    node_pointer = {}
    pointer = 2
    #
    # (c) Optional graph of the linearised graph; original nodes are
    # in green
    #
    if graph_output:
        stretched_graph = pydot.Dot(graph_type='digraph')
    #
    # List of available nodes for finding a match of
    # channels. Elements of this list will be pairs, consisting of the
    # node (name) 'n' together with its name node_name(n,i) in the
    # graph. Since nodes may occur multiple times, we need to use
    # different names in the graph.
    #
    available_nodes = []
    #
    # Step 1: handling of initial nodes; 
    #
    # deep copy
    unprocessed_nodes = [n for n in nodes]
    initial_nodes = []
    initial_copy_channels = []
    for n in nodes:
        # print("Handling node: ", n)
        parents_n = parents[n]
        if len(parents_n) == 0:
            # n is an initial node
            initial_nodes.append(n)
            unprocessed_nodes.remove(n)
            node_pointer[n] = 0          
            if graph_output:
                stretched_graph.add_node(pydot.Node(n, 
                                                    style="filled", 
                                                    fillcolor="green"))
            if observed:
                available_nodes.append(n)
            children_n = children[n] 
            children_num = len(children_n)
            # print("Children of: ", n, children_n)
            #
            # Make copies of the initial states, depending on the
            # number of children (and on whether or not we want an
            # 'observed' version)
            #
            if children_num > 0:
                if graph_output:
                    stretched_graph.add_node(pydot.Node(n + "!copy", 
                                                        width=0.15,
                                                        height=0.15,
                                                        fixedsize=True,
                                                        style="filled", 
                                                        fillcolor="black",
                                                        fontsize=0))
                    stretched_graph.add_edge(pydot.Edge(n, n + "!copy"))
                    if observed:
                        stretched_graph.add_edge(pydot.Edge(n + "!copy",
                                                            " " + n + " "))
                available_nodes += children_num * [n]
                initial_copy_channels.append(copy(channels[n].dom, 
                                                  children_num + 
                                                  (1 if observed else 0)))
    if len(initial_nodes) == 0:
        raise Exception('Error: the model does not have initial nodes')
    state = reduce(lambda s1, s2: s1 @ s2, [channels[n]
                                            for n in initial_nodes])
    channel_list.append(state)
    if len(initial_copy_channels) == 0:
        # Initial nodes have no children; extremely simple graph
        channel_list.append(idn(state))
    else:
        initial_copy_chan = reduce(lambda c1, c2: c1 @ c2, 
                                   initial_copy_channels)
        channel_list.append(initial_copy_chan)
    # print("Initial state: ", channel_list[1] >> state )
    #
    # Step 2: continue with non-initial nodes from the model: cycle
    # through the set of unprocessed nodes until all associated
    # channels have been applied to the current state, and the list of
    # unprocessed nodes is empty.
    #
    iterations = 0
    while len(unprocessed_nodes) > 0:
        for un in unprocessed_nodes:
            iterations += 1
            un_chan = channels[un]
            parents_un = [n.name for n in un_chan.dom.names]
            num_parents_un = len(parents_un)
            current_dom = channel_list[len(channel_list)-1].cod
            dom_lens = [len(d) for d in current_dom]
            print("* Iteration", iterations, "for", un, "of size:", dom_lens,
                  "is", reduce(operator.mul, dom_lens, 1))
            search_copy_of_nodes = [u for u in available_nodes]
            len_available_nodes = len(available_nodes)
            # print("Parents of: ", un, parents_un, "searching in", available_nodes)
            swaps = list(range(len_available_nodes))
            #
            # find occurrences of un's parents in domains
            #
            i = 0
            found_all = True
            while i < num_parents_un:
                # print("... searching for parent: ", parents_un[i] )
                #
                # try to find i-th parent among domains
                #
                j = 0
                found_i = False
                while j < len_available_nodes:
                    if search_copy_of_nodes[j] == parents_un[i]:
                        found_i = True
                        break
                    j += 1
                if not found_i:
                    #
                    # stop handling parent i
                    #
                    found_all = False
                    # print("Stop handling node: ", un)
                    break
                # print("=> Parent found of:", un, "=", parents_un[i], "at", j)
                #
                # i-th parent found at j; now swap j |-> i
                #
                tmp = swaps[j]
                swaps[j] = swaps[i]
                swaps[i] = tmp
                search_copy_of_nodes[j] = search_copy_of_nodes[i]
                search_copy_of_nodes[i] = parents_un[i]
                i += 1
            if found_all:
                #
                # all parents found; now update the state with channel of un
                #
                print("==> All parents found of:", un)
                # print("Available domains: ", available_nodes)
                if graph_output:
                    stretched_graph.add_node(pydot.Node(un, 
                                                        style="filled", 
                                                        fillcolor="green"))
                #
                # incorporate swaps into available nodes and arguments
                #
                inv_swaps = len_available_nodes * [0]
                swapped_doms = []
                swapped_available_nodes = []
                for i in range(len_available_nodes):
                    inv_swaps[swaps[i]] = i
                    swapped_doms.append(current_dom.get_nameditem(
                        #argument_swaps[i]))
                        swaps[i]))
                    swapped_available_nodes.append(available_nodes[swaps[i]])
                swapped_dom = reduce(lambda d1, d2: d1 + d2, swapped_doms)
                # print("Swaps:", swaps, inv_swaps )
                #
                # Build the channel that does the swapping
                #
                available_nodes = swapped_available_nodes
                # print("Swapped domains: ", available_nodes)
                if graph_output:
                    for i in range(num_parents_un):
                        stretched_graph.add_edge(pydot.Edge(
                            available_nodes[i] + "!copy", un))
                un_chan_id = un_chan
                diff = len_available_nodes - num_parents_un
                if diff > 0:
                    identities_doms = []
                    for i in range(diff):
                        d_i = swapped_dom.get_nameditem(i + num_parents_un)
                        identities_doms.append(d_i)
                    identities_dom = reduce(lambda d1, d2: d1 + d2, 
                                            identities_doms)
                    identities = idn(identities_dom)
                    un_chan_id = un_chan @ identities
                #
                # Add the channel to the list, with its domains permuted
                # 
                channel_list.append(perm_chan(un_chan_id, 
                                              dom_perm = inv_swaps))
                pointer += 1
                node_pointer[un] = pointer
                #
                # Update the available nodes
                #
                tails = available_nodes[num_parents_un:]
                heads = [un]
                childred_un = children[un] 
                num_children_un = len(childred_un)
                # print("Children of: ", un, childred_un)
                if num_children_un > 0:
                    if graph_output:
                        stretched_graph.add_node(pydot.Node(un + "!copy", 
                                                            width=0.15,
                                                            height=0.15,
                                                            fixedsize=True,
                                                            style="filled", 
                                                            fillcolor="black",
                                                            fontsize=0))
                        stretched_graph.add_edge(pydot.Edge(un, un + "!copy"))
                        if observed:
                            stretched_graph.add_edge(pydot.Edge(un + "!copy",
                                                                " " + un + " "))
                    if num_children_un > 1 or observed:
                        heads += (num_children_un - 1 + 
                                  (1 if observed else 0)) * [un]
                        lcs = len(channel_list)
                        c = channel_list[lcs-1]
                        ls = len(c.cod) * [1]
                        ls[0] = num_children_un + (1 if observed else 0)
                        channel_list[lcs-1] = copy_chan(c, ls)
                available_nodes = heads + tails
                #print("Finished copying")
                #
                # Terminate handling node un and return to for-loop
                #
                unprocessed_nodes.remove(un)
    #
    # Collect the results in a dictionary
    # 
    result = {
        'pointer'  : node_pointer,
        'channels' : channel_list
    }
    if graph_output:
        result['graph'] = stretched_graph
    return result

"""
