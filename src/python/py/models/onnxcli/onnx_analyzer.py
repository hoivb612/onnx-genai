import onnx
import argparse

def extract_subgraphs(graph):
    """
    Recursively extract all subgraphs from a given GraphProto.
    This collects the main graph and any nested graphs found in node attributes.
    """
    subgraphs = []

    def traverse(g):
        subgraphs.append(g)
        # Look for subgraphs in node attributes (e.g., inside control flow nodes like If/Loop).
        for node in g.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    traverse(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for sub_g in attr.graphs:
                        traverse(sub_g)
    traverse(graph)
    return subgraphs

def extract_consecutive_supported_sequences(graph, supported_ops, min_length=2):
    """
    Walk through the nodes in the graph and collect sequences of consecutive nodes
    (as they appear in the node list) that have supported operators.
    
    Args:
        graph (onnx.GraphProto): The graph to process.
        supported_ops (set): A set of operator names (strings) considered supported.
        min_length (int): The minimum number of consecutive supported nodes required to record a sequence.
        
    Returns:
        A list of sequences, where each sequence is a list of operator names.
    """
    sequences = []
    current_sequence = []
    for node in graph.node:
        if node.op_type in supported_ops:
            current_sequence.append(node.op_type)
        else:
            if len(current_sequence) >= min_length:
                sequences.append(current_sequence)
            current_sequence = []
    # If the last nodes in the graph were supported, capture that sequence.
    if len(current_sequence) >= min_length:
        sequences.append(current_sequence)
    return sequences

def collate_unique_quantizer_ops(found_sequences):
    """
    Collates all supported sequences from every subgraph and returns a sorted list
    of unique operator names that are recommended to be fed to a quantizer.
    
    Args:
        found_sequences: A list of tuples of the form (graph_name, list_of_sequences),
                         where each sequence is a list of operator names.
                         
    Returns:
        A sorted list of unique operator names.
    """
    unique_ops = set()
    for graph_name, sequences in found_sequences:
        for seq in sequences:
            for op in seq:
                unique_ops.add(op)
    return sorted(unique_ops)

def load_and_format_operator_file(file_path):
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Strip newline characters and create a set of strings
    supported_ops = {line.strip() for line in lines}
    
    return supported_ops



def main():
    parser = argparse.ArgumentParser(
        description="Load an ONNX model, detect sequences of consecutive supported operators in subgraphs, "
                    "and return unique operators recommended for quantization.  This list may not work out-of-the-box, as some operators may be supported on the NPU but the order or combination may not.  Start with this list of operators and remove operators until you see NPU offload and then add each removed operator back into the list- this will likely result in more subgraphs being loaded to NPU improving your overall performance."
    )
    parser.add_argument("--onnx_file", action="store", required=True, help="Path to the ONNX file")
    parser.add_argument("--supported_operators", action="store", type=str, help="Name of the window you wish to run this app against.", default="onnx_operators.txt")
    args = parser.parse_args()

    # Define your supported operators (adjust as needed).
    #VitisAI Supported Operations
    supported_ops = load_and_format_operator_file("onnx_operators.txt")

    # Load the ONNX model.
    try:
        model = onnx.load(args.onnx_file)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # iterate through inputs of the graph
    for input in model.graph.input:
        print (input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                     print (d.dim_value, end=", ")  # known dimension
                elif (d.HasField("dim_param")):
                     print (d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print ("?", end=", ")  # unknown dimension with no name
        else:
            print ("unknown rank", end="")
        print()

"""
    # Extract all subgraphs (main graph and any nested ones).
    all_subgraphs = extract_subgraphs(model.graph)

    # List to store tuples: (subgraph_name, list_of_supported_sequences)
    found_sequences = []
    for idx, subg in enumerate(all_subgraphs):
        # Use the graph's name if available; otherwise, generate one based on the index.
        graph_name = subg.name if subg.name else f"Graph_{idx}"
        sequences = extract_consecutive_supported_sequences(subg, supported_ops, min_length=2)
        if sequences:
            found_sequences.append((graph_name, sequences))

    # Report sequences found per subgraph.
    # Do not need to show this since we really care about the final list.
    if found_sequences:
        print("Subgraphs with consecutive supported operator sequences:")
        for name, seq_list in found_sequences:
            print(f"\nSubgraph: {name}")
            for seq in seq_list:
                print(f"  Supported sequence: {seq}")
    else:
        print("No subgraphs found with consecutive supported operator sequences.")
    
    # Collate unique operators from all supported sequences.
    unique_ops = collate_unique_quantizer_ops(found_sequences)
    if unique_ops:
        print("\nRecommended unique operators for quantization:")
        print(unique_ops)
    else:
        print("\nNo unique supported operators found for quantization.")
"""

if __name__ == "__main__":
    main()


