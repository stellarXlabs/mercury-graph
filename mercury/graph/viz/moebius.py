import importlib.util, inspect, json, os

import pandas as pd

from mercury.graph.core.spark_interface import SparkInterface

HTML, Javascript, display = None, None, None

if importlib.util.find_spec('IPython.display') is not None:
    from IPython.display import HTML, Javascript, display


class Moebius:
    '''
    Moebius class for visualizing graphs using JavaScript and HTML.

    Usage:
        ```python
        from mercury.graph.viz import Moebius

        G = ... # A graph object
        moebius = Moebius(G)
        moebius.show()
        ```

    Attributes:
        G (Graph):          The graph to be visualized.
        use_spark (bool):   Flag indicating if Spark is used.
        front_pat (str):    Path to the frontend resources.
        _int_id_map (dict): A dictionary mapping node IDs to integer IDs.
        name():             The instance name of the object required by the JS callback mechanism.
    '''

    def __init__(self, G):
        if HTML is None:
            raise ImportError('IPython is not installed')

        self.G = G
        self.use_spark = self.G._as_networkx is None
        self.front_pat = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/frontend'
        self._int_id_map = {node['id'] : i for i, node in enumerate(self.G.nodes)}


    def __str__(self):
        return 'Moebius object %s' % self.name


    def __getitem__(self, item):
        return self._get_adjacent_nodes_moebius(item)


    @property
    def name(self):
        ''' Get the instance name of the object which is required by the JS callback mechanism. '''
        return self._get_instance_name()


    def JS(self, s):
        ''' Syntactic sugar for display(Javascript()) '''
        display(Javascript(s))


    def FJS(self, fn):
        ''' Syntactic sugar for display(Javascript(filename = fn)) '''
        display(Javascript(filename = fn))


    def FHT(self, fn):
        ''' Syntactic sugar for display(HTML(filename = fn)) '''
        display(HTML(filename = fn))


    def edge_config(self, label = None, size = None, size_range = None, size_scale = 'linear', color_palette = None):
        '''
        Create an edge configuration dictionary for show() in an understandable way.

        Args:
            label (str): The edge attribute to be displayed as text over the edges.
            size (str): The edge attribute to be displayed as the width of the edges. See the options in the Moebius configuration
                menu to set limits and scales.
            size_range (List of two numbers): Combined with label, this parameter controls the values in the variable that
                correspond to the minimum and maximum displayed width. The values below or equal the first value will be displayed with
                the base width (that depends on the zoom) and the values above or equal to the second value will be shown with the maximum
                width.
            size_scale ('linear', 'power', 'sqrt' or 'log'): Combined with label, the scale used to convert the value in the
                variable to the displayed width.

        Returns:
            The edge configuration dictionary
        '''
        config = {}

        if color_palette is not None:
            config['color_palette'] = color_palette
        else:
            config['color_palette'] = {}

        if label is not None:
            config['label'] = label

        if size is not None:
            config['sizes_col'] = size

            if size_range is None:
                config['size_thresholds'] = []
            else:
                config['size_thresholds'] = size_range

            if size_scale in ['linear', 'power', 'sqrt', 'log']:
                config['scale'] = size_scale
        else:
             config['size_thresholds'] = []

        return config


    def node_config(self, label = None, category = None, colors = None, size = None, range = None, scale = 'linear'):
        '''
        Create a node configuration dictionary for show() in an understandable way.

        Args:
            label (str): The node attribute to be displayed as text over the nodes. Use the string `ìd` to draw the node id (regardless
                of the column having another name) or any valid node attribute name.
            category (str): A categorical node attribute that can be represented as the node color. This will also enable a legend
                interface where categories can be individually shown or hidden.
            colors (dict): The colors for each category defined as a dictionary. The keys are possible outcomes of category.
                The values are html RGB strings. E.g., .draw(category = 'size', colors = {'big' : '#c0a080', 'small' : '#a0c080'})
                where 'big' and 'small' are possible values of the category 'size'.
            size (str): The node attribute to be displayed as the size of the nodes. Use the string `ìd` to set the node id (regardless
                of the column having another name) or any valid node attribute name. See the options in the Moebius configuration menu to
                set minimum, maximum sizes, linear or logarithmic scale, etc.
            range (List of two numbers): Combined with edge_label, this parameter controls the values in the variable that
                correspond to the minimum and maximum displayed sizes. The values below or equal the first value will be displayed with the
                base radius (that depends on the zoom) and the values above or equal to the second value will be shown with the maximum
                radius.
            scale ('linear', 'power', 'sqrt' or 'log'): Combined with edge_label, the scale used to convert the value in the variable to
                the displayed radius.

        Returns:
            The node configuration dictionary
        '''
        config = {}

        if label is not None:
            config['label'] = label

        if category is not None:
            config['color'] = category

        if colors is not None:
            config['color_palette'] = colors
        else:
            config['color_palette'] = {}

        if size is not None:
            config['sizes_col'] = size

            if range is None:
                config['size_thresholds'] = []
            else:
                config['size_thresholds'] = range

            if scale in ['linear', 'power', 'sqrt', 'log']:
                config['scale'] = scale
        else:
            config['size_thresholds'] = []

        return config


    def show(self, initial_id = None, initial_depth = 1, node_config = None, edge_config = None):
        '''
        Start the interactive graph visualization in a Jupyter notebook.

        Args:
            initial_id (str): The id of the node to start the visualization.
            initial_depth (int): The initial depth of the graph (starting with `initial_id` as 0) to be shown.
            node_config (dict): A node configuration dictionary created by `node_config()`.
            edge_config (dict): An edge configuration dictionary created by `edge_config()`.
        '''
        if initial_id is None:
            initial_id = self._int_id_map.keys()[0]

        initial_json = self._get_adjacent_nodes_moebius(initial_id, depth = initial_depth)

        if node_config is None:
            node_config = self.node_config()

        if edge_config is None:
            edge_config = self.edge_config()

        self._load_moebius_js(initial_json, self.name, node_config, edge_config)


    def _get_instance_name(self):
        '''
        Get the instance name of the object
        '''
        ll = [k for k, v in globals().items() if v is self]
        if ll:
            return ll[0]

        main = importlib.import_module('__main__')
        ll = [k for k, v in vars(main).items() if v is self]
        if ll:
            return ll[0]

        frame = inspect.currentframe()
        ll = [k for k, v in frame.f_back.f_locals.items()
              if v is self and k != 'self']
        if ll:
            return ll[0]

        ll = [k for k, v in frame.f_back.f_back.f_locals.items()
              if v is self and k != 'self']
        if ll:
            return ll[0]

        raise NotImplemented('Could not find instance name')


    def _load_moebius_js(self, initial_json, instance_name, node_config, edge_config):
        '''
        Load the Moebius javascript library and call the function to draw the graph
        '''
        self.JS('require.config({paths: {d3: \'https://d3js.org/d3.v4.min\'}});')
        self.FJS(self.front_pat + '/moebius.js')
        self.FHT(self.front_pat + '/moebius.svg.html')
        self.FHT(self.front_pat + '/moebius.css.html')

        javascript_moebius_call = '''
            (function(element) {
                require(['moebius'], function(moebius) {
                    moebius(%s, '%s', %s, %s);
                });
            })(element);
        ''' % (initial_json, instance_name, node_config, edge_config)

        self.JS(javascript_moebius_call)


    def _get_adjacent_nodes_moebius(self, node_id, limit = 20, depth = 1):
        '''
        This is the callback function used to interact with the Moebius JS library. Each time you deploy a node in the graph, this
        function is called to get the adjacent nodes and edges.

        The expected return is a JSON string with the following format:

        json = {
            'nodes': [
                {'id' : ..., 'count' : ..., '_int_id' : ...},
                {'id' : ..., 'count' : ..., '_int_id' : ...}
            ],
            'links': [
                {'source' : ..., 'target' : ..., '_int_id' : ...},
                {'source' : ..., 'target' : ..., '_int_id' : ...}
            ]
        },
        where 'count' is the degree of the given node, and '_int_id' is a
        unique integer identifier for the given node or edge.

        Args:
            node_id (str): The id of the node to get the adjacent nodes.
            limit (int): The maximum number of nodes to be returned.
            depth (int): The depth of the graph to be returned.

        Returns:
            A JSON string with the adjacent nodes and edges.
        '''

        if self.use_spark:
            nodes_df, edges_df = self._get_one_level_subgraph_graphframes(node_id)
            N = nodes_df.count()

        else:
            nodes_df, edges_df = self._get_one_level_subgraph_networkx(node_id)
            N = len(nodes_df)

        d = 1

        while N < limit and d < depth:
            pass

        if self.use_spark:
            pass

        else:
            json_final = {
                'nodes': json.loads(nodes_df.to_json(orient = 'records')),
                'links': json.loads(edges_df.to_json(orient = 'records'))
            }

        return json.dumps(json_final)


    def _get_one_level_subgraph_graphframes(self, node_id):
        '''
        Get the one-level subgraph for a given node ID using GraphFrames.

        Args:
            node_id (str): The ID of the node for which to get the one-level subgraph.

        Returns:
            tuple: A tuple containing two Spark DataFrames:
                - nodes_df: DataFrame with columns 'id', 'count', '_int_id', and any other node attributes.
                - edges_df: DataFrame with the edges connecting the nodes in the subgraph.
        '''
        sql         = SparkInterface().pyspark.sql
        F           = sql.functions
        IntegerType = sql.types.IntegerType

        graph = self.graphframe

        neighbors = graph.bfs(fromExpr = 'id = ' % node_id, toExpr = 'true', maxPathLength = 1)
        nodes_df  = neighbors.select('from.*').distinct()

        degrees  = graph.degrees
        nodes_df = nodes_df.join(degrees, on = 'id', how = 'left').withColumnRenamed('degree', 'count')

        # Define a local UDF to map id to _int_id using the dictionary
        def get_int_id(node_id):
            return self._int_id_map[node_id]

        # Register the UDF
        get_int_id_udf = F.udf(get_int_id, IntegerType())

        # Add the _int_id column to nodes_df
        nodes_df = nodes_df.withColumn('_int_id', get_int_id_udf(F.col('id')))

        edges_df = graph.edges.filter(
            (graph.edges.src.isin(nodes_df.select('id').rdd.flatMap(lambda x: x).collect())) &
            (graph.edges.dst.isin(nodes_df.select('id').rdd.flatMap(lambda x: x).collect()))
        )

        return nodes_df, edges_df


    def _get_one_level_subgraph_networkx(self, node_id):
        '''
        Get the one-level subgraph for a given node ID using Networkx.

        Args:
            node_id (str): The ID of the node for which to get the one-level subgraph.

        Returns:
            tuple: A tuple containing two Spark DataFrames:
                - nodes_df: DataFrame with columns 'id', 'count', '_int_id', and any other node attributes.
                - edges_df: DataFrame with the edges connecting the nodes in the subgraph.
        '''
        graph = self.G.networkx

        neighbors = list(graph.neighbors(node_id)) + [node_id]
        subgraph  = graph.subgraph(neighbors)

        # Create nodes DataFrame
        nodes_data = []
        for node in subgraph.nodes(data=True):
            node_id = node[0]
            attributes = node[1]
            attributes['id'] = node_id
            attributes['count'] = graph.degree[node_id]
            attributes['_int_id'] = self._int_id_map[node_id]
            nodes_data.append(attributes)
        nodes_df = pd.DataFrame(nodes_data)

        # Create edges DataFrame
        edges_data = []
        for edge in subgraph.edges(data=True):
            src, dst, attributes = edge
            attributes['src'] = src
            attributes['dst'] = dst
            edges_data.append(attributes)
        edges_df = pd.DataFrame(edges_data)

        return nodes_df, edges_df


    def _pd_to_json_format(self, df):
        '''
        A utility to produced the flavor of JSON expected by the JS library from a pandas DataFrame.

        Usage:
            ```python
            # This simple snippet to convert the whole graph to the expected JSON format.

            edges = self._pd_to_json_format(self.G.edges_as_pandas().rename({'src' : 'source', 'dst' : 'target'}, axis = 1))
            nodes = self._pd_to_json_format(self.G.nodes_as_pandas())

            json_final = {'nodes' : nodes, 'links' : edges}

            return json.dumps(json_final, ensure_ascii=False)
            ```

        Args:
            df (pd.DataFrame): The DataFrame to be converted to JSON.
        '''

        df = df.reset_index(drop = True)

        df_json = df.to_json(orient = 'records', force_ascii = True, date_format = 'iso')

        return json.loads(df_json)
