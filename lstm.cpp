/*									lstm.cpp
 * 							//full blown code for lstm
 * 							//dont require any other  library or toolkit to run the code
 * 							//written by Punnoose A K
 * 
 * 
 * 
 * 
 */ 




#include <time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bits/stdc++.h>

using namespace std;





typedef double floater;

//size of the input vector
 

//size of the cell state  
#define M 3 

//both M and O shoudld be the same

//size of the hidden state
#define O 3

//num of hidden layers in each fully connected  units
#define NUM_HIDDEN_LAYERS 3

// num of data points to be taken
#define BATCH_SIZE 100


//data related variables
//each row is an x-y time series 
#define MAX_DATA_ROWS 100000				//amx possible rows. each row corresoponds to a <x,y> time series

//max data line width of a single line in the data file. A single line corresponds to an <x,y> time series
#define MAX_DATA_LINE_WIDTH 10000


//max width for a single value in the data file. eg 1.2323222e-233
#define MAX_VAL_WIDTH 100

//maximum (x,y) pairs in 1 time-series in 1 row
#define MAX_TIME_INSTANCES 10000 					


//hidden node size. all the hidden nodes would be of this size
#define H  3


//this is used for neural network width. this must be greater than max size of the hidden nodes and the input and the output dimensions
#define MAX_NODE_SIZE 1000




//dimension of input data
int global_data_x_dim;

//dimension of y
int global_data_y_dim;

//number of rows in the whole data file
int global_data_num_rows;


//number of (x-y) pairs in a time series in 1 row
int global_data_time_instances;


int N;// must be equal to global_data_x_dim

const double LEARNING_RATE=0.05;

floater ***global_data_x;
floater ***global_data_y;




//vect class with all the operators
class vect
{
	public:
	//size 
	int v_size;															
	floater *v;
	void display_vector();								//displays the vector
	
};





//function definitions
vect *add_vectors(vect *, vect *);
vect *sub_vectors(vect *, vect *);
floater dot_product(vect *, vect *);
vect *concat_vector(vect *, vect *);
vect *pointwise_product(vect *, vect *);
floater sigmoid(floater );

void parse_line(char [], int);
void read_data(char *, int, int, int );

void extract(char [], int , int, bool, int);
vect *create_vector_array(int , bool);

void print_data();

floater softmax(vect *, int);
floater cross_entropy(vect *, vect *);


floater derivative_sigmoid(floater);
floater derivative_tanh(floater);
floater derivative_softmax(vect *, int , int );


vect *vect_to_softmax(vect *);


void destroy_vector_array(vect *);



//this is just a simple 2 layer fully connected layer with just softmax and weights
class fully_connected_layer
{
	
	public:
	int *w_size;			//only 2 layers							
	floater *b;					//2 biases	
	floater **out;				//2 layers 
	floater **in;				//2 layers	
	floater **W;				//2 layers
	
	vect *differential_outgoing;	// gets populated by the backprogation routinr
	
	vect *forward_pass(vect *);		//forward pass of a vector and returns a vector as the output
	void backpropogation(vect *);	//this will populate the differntial_outgoing vector, from an incoming loss vector
	void random_initialize();		//intiailze the weights to random values
	
};




//represents the parameters of a whole sigmoid fully connected layer. every node is sigmoid. 
class sigmoid_layer
{
	
	
	public:
	floater ***W; //3d weights
	int *w_size;  //size of  all layers including input +hidden + output w[0] is the input layer
	floater *b;   //b[0] input bias
	int hidden_num_layers; // only number of hidden layers. 
	floater **out;	//to represent the output of each sigmoid. out[0][0] is the output of the first node, first layer
	floater **in; 	//to represent the net input to each sigmoid
	
	vect *differential_outgoing;			//outgoing differential from an incming loss vector
	
	vect *forward_pass(vect *);				//this will run a vector through the sigmoid layer and returns a vector as output
	void random_initialize();				// initialze the weights and  bias to random values
	void print_weights();					//print the weights of the sigmoid layer
	void print_bias();						//print the bias of the sigmoid layer
	void print_outs();						//print the outs of the sigmoid layer
	void print_ins();						//print the ins
	void backpropogation(vect *);			//do the  backpropogation from an incoming loss vector. this will populate the differential_outgoing
};

//represents the parameters of the sigmoid layer
class tanh_layer
{
	
	
	public:
	floater ***W;
	int *w_size;
	floater *b;
	int hidden_num_layers;
	floater **out;
	floater **in;
	
	vect *differential_outgoing;			//outgoing differential from an incming loss vector
	
	vect *forward_pass(vect *);				//this will run a vector through the tanh layer and returns a vector as output
	
	
	void random_initialize();				//initialize the weights with random values
	
	
	void print_weights();					//print the weights of the tanh layer
	void print_bias();						//print the bias of the tanh layer
	void print_outs();						//print the outs of the tanh layer
	void print_ins();						// print the ins of the tanh layer
	
	void backpropogation(vect *);			//do the backpropogation from an incoming loss vector and populate differential_outgoing
	
};




	
	
//represents an lstm cell
class cell
{
	
	public:
	
	//inputs from under
	vect *input;
	
	
	//coming from the left
	vect *previous_hidden_output;
	vect *previous_cellstate_output;
	
	//outputs. going to the right
	vect *current_hidden_output;
	vect *current_cellstate_output;
	
	vect *current_output;  //this is the same as that of hidden output 		
	
	
	//intermediate vectors
	vect *combined;						//combined vector as input
	vect *fs;							//output of forget_sigmoid gate	
	vect *is;							//output of input_sigmoid
	vect *it;							//output of input_tanh
	vect *ist;							//output after input pointwise multiplication
	vect *fsc;							//output after input pointwise multiplcation of fs and cell state
	vect *fscist;						//output after addition of fsc and ist. this ios the cell state
	vect *os;							//output after output_sigmoid
	vect *ot;							//output after output_tanh
	vect *ost;							//output after pointwise multiplication of os and ost
	vect *ost_reduced;					//output after passing ost through the final fully connected layer
										//ost_reduced is the softmaxed version probability vector
	
	
	//differential of all the above vectors
	vect *differential_combined;						//combined vector as input
	vect *differential_fs;							//differential wrt output of forget_sigmoid gate	
	vect *differential_is;							//differential wrt output of input_sigmoid
	vect *differential_it;							//differential wrt output of input_tanh
	vect *differential_ist;							//differential wrt output after input pointwise multiplication
	vect *differential_fsc;							//differential wrt output after input pointwise multiplcation of fs and cell state
	vect *differential_fscist;						//differential wrt output after addition of fsc and ist. this ios the cell state
	vect *differential_os;							//differential wrt output after output_sigmoid
	vect *differential_ot;							//differential wrt output after output_tanh
	vect *differential_ost;							//differential wrt output after pointwise multiplication of os and ost
	
	
	vect *differential_bfs;							//differntial coming back to before the forget_sigmoid
	vect *differential_bis;						//differntial coming back to before the input_sigmoid
	vect *differential_bit;                         //differntial coming back to before the input_tanh
	vect *differential_bos;							//differntial coming back to before the output sigmoid
		
	//coming from the right
	vect *incoming_cellstate_differential;
	vect *incoming_hidden_output_differential;
	
	//outgoing to the left
	vect *current_cellstate_differential;
	vect *current_hidden_output_differential;
	
	
	//define the different  gate weights
	sigmoid_layer *forget_sigmoid, *input_sigmoid, *output_sigmoid;
	tanh_layer *input_tanh, *output_tanh;
	fully_connected_layer *final_fully_connected_layer;				
	
	//this will initilaize the parameters
	void initialize_parameters();
	
	//compute all the outputs from the inputs
	void compute_outputs(vect *);
	
	//this will pritn the output of a single cell
	void print_outputs();
	
	
	//dealocate the sigmoid and tanh parameters
	void deallocate();
	
	//do the full blown backpropogation taking the succeeding cell_state, and the succeeding hidden_state
	//onoy cuurent target has to be supplied
	void cell_backpropogation(vect *); 
	
	//print all the vectors in forward pass
	void print_intermediate_vectors_forward_pass(); 
	
	//print all the intermediate vectors in the backward pass
	void print_intermediate_vector_backward_pass();
	
};	
	


//main class of lstm
class lstm
{
	
	public:
	int num_cells;					//number of cells in the lstm
	cell *cell_array;				//a 1d array of cells;
	
	void create_cell_array(int);    //create the cell_array with the specified number of cells.
	void forward_pass(vect *);       //forward pass with  a vect array(with size global data time instances) through the lstm
	
	void print_lstm_outputs();		//print the output of every cells
	void backward_pass(vect *);		//core backward pass. takes input the vect array with size global data time instances	
	
	
	void destroy_cell_array();		// destroy all the cells in the cell array
};



//takes a vect_array with global data time instance
//supply each cell with the y data
//start with an empty incoming diffferentials for the cell state and the hidden state and pass the outgoing differentials to the
//preceeding cells
//this routine should be called from the main routine
void lstm::backward_pass(vect *vect_array)
{
	
	
	int i,j;
	//create an empty incoming hidden output differntial  and incoming currentstate differential
	vect *temp_incoming_cellstate_differntial;
	vect *temp_incoming_hidden_output_differential;
	
	temp_incoming_hidden_output_differential=(vect *)malloc(sizeof(vect));
	temp_incoming_hidden_output_differential->v_size=O;
	temp_incoming_hidden_output_differential->v=(floater *)malloc(O*sizeof(floater));
	for(i=0;i<O;i++)
	{
		temp_incoming_hidden_output_differential->v[i]=0;
	}
	
	temp_incoming_cellstate_differntial=(vect *)malloc(sizeof(vect));
	temp_incoming_cellstate_differntial->v_size=M;
	temp_incoming_cellstate_differntial->v=(floater *)malloc(M*sizeof(floater));
	for(i=0;i<M;i++)
	{
		temp_incoming_cellstate_differntial->v[i]=0;
	}
	
	
	
	//supply to the last cell the empty incoming differntials
	//call the cell_backporpogation with last vect in the vect array as the input
	cell_array[global_data_time_instances-1].incoming_cellstate_differential=temp_incoming_cellstate_differntial;
	cell_array[global_data_time_instances-1].incoming_hidden_output_differential=temp_incoming_hidden_output_differential;
	cell_array[global_data_time_instances-1].cell_backpropogation(&vect_array[global_data_time_instances-1]);
	
	
	//vect_array->display_vector();
	
	
	//do the backpropogation for all the cells
	for(j=global_data_time_instances-2;j>=0;j--)
	{
		
		cell_array[j].incoming_cellstate_differential=cell_array[j+1].current_cellstate_differential;
		cell_array[j].incoming_hidden_output_differential=cell_array[j+1].current_hidden_output_differential;
		cell_array[j].cell_backpropogation(&vect_array[j]);
		
	}
	
		
}//end of backward pass





//print the output of all the cells of lstm
void lstm::print_lstm_outputs()
{
	
	int i;
	for(i=0;i<num_cells;i++)
	{
		
		cell_array[i].print_outputs();
		cout<<"\n";
	}
	
	
}




//this will create the cells required for the lstm
void lstm::create_cell_array(int num_cell)
{
	
	int i;
	cell_array=(cell *)malloc(num_cell*sizeof(cell));
	num_cells=num_cell;
	//initialize all the cells
	for(i=0;i<num_cells;i++)
	{
		cell_array[i].initialize_parameters();
	}
	
	
	
	
}



//clean up routine. must be called last
void lstm::destroy_cell_array()
{
	
	int i;
	for(i=0;i<num_cells;i++)
	{
		//deallocate every variable associated with the cell
		//W, b, in, out, fs,fo,....differential_fs,differential_fo
		cell_array[i].deallocate(); 
	}
	free(cell_array);
	
	
}


//do the forward pass with the vect_array across the lstm cells
void lstm::forward_pass(vect *vect_array)
{
	
	int i;
	vect *temp;
	
	//initialize the previosu cell state  of previous hidden states
	cell_array[0].previous_cellstate_output=(vect *)malloc(sizeof(vect));
	cell_array[0].previous_cellstate_output->v_size=M;
	cell_array[0].previous_cellstate_output->v=(floater *)malloc(M*sizeof(floater));
	
	
	//initialize the previous_cellstate_output of the  first cell to 0
	for(i=0;i<M;i++)
	{
		cell_array[0].previous_cellstate_output->v[i]=0;
	}
	
	
	//initialize the previous hidden output of cell[0] to 0
	cell_array[0].previous_hidden_output=(vect *)malloc(sizeof(vect));
	cell_array[0].previous_hidden_output->v_size=O;
	cell_array[0].previous_hidden_output->v=(floater *)malloc(O*sizeof(floater));
	for(i=0;i<O;i++)
	{
		cell_array[0].previous_hidden_output->v[i]=0;
	}
	
	
	temp=&vect_array[0];
	//this will compute the outputs of cell[0]
	cell_array[0].compute_outputs(temp);
	
	
	//pass the output  of the previous cell to the next cell and compute the output
	for(i=1;i<global_data_time_instances;i++)
	{
		temp=&vect_array[i];
		
		
		cell_array[i].previous_hidden_output=cell_array[i-1].current_hidden_output;
		cell_array[i].previous_cellstate_output=cell_array[i-1].current_cellstate_output;
		
		//call the compute
		cell_array[i].compute_outputs(temp);
	}	
	
		
}





//deallocate all the memory assingned for all the weights
void cell::deallocate()
{
	
	int i,j,k;
	
	//free forget_sigmoid
	for(i=0;i<forget_sigmoid->hidden_num_layers+1;i++)
	{
		for(j=0;j<forget_sigmoid->w_size[i];j++)
		{
			free(forget_sigmoid->W[i][j]);
		}
		free(forget_sigmoid->W[i]);
			
	}
	free(forget_sigmoid->b);
	
	for(i=0;i<forget_sigmoid->hidden_num_layers+2;i++)
	{
		free(forget_sigmoid->out[i]);
		free(forget_sigmoid->in[i]);
	}
	free(forget_sigmoid->out);
	free(forget_sigmoid->in);
	free(forget_sigmoid->w_size);
	
	
	
	//free input_sigmoid
	for(i=0;i<input_sigmoid->hidden_num_layers+1;i++)
	{
		for(j=0;j<input_sigmoid->w_size[i];j++)
		{
			free(input_sigmoid->W[i][j]);
		}
		free(input_sigmoid->W[i]);
		
	}
	free(input_sigmoid->b);
	
	for(i=0;i<input_sigmoid->hidden_num_layers+2;i++)
	{
		free(input_sigmoid->out[i]);
		free(input_sigmoid->in[i]);
	}
	free(input_sigmoid->out);
	free(input_sigmoid->in);
	free(input_sigmoid->w_size);
	
	
	
	
	//free output sigmoid
	for(i=0;i<output_sigmoid->hidden_num_layers+1;i++)
	{
		for(j=0;j<output_sigmoid->w_size[i];j++)
		{
			free(output_sigmoid->W[i][j]);
		}
		free(output_sigmoid->W[i]);
		
	}
	free(output_sigmoid->b);
	
	
	for(i=0;i<output_sigmoid->hidden_num_layers+2;i++)
	{
		free(output_sigmoid->out[i]);
		free(output_sigmoid->in[i]);
	}
	free(output_sigmoid->out);
	free(output_sigmoid->in);
	free(output_sigmoid->w_size);
	
	
	
	//free input tanh
	for(i=0;i<input_tanh->hidden_num_layers+1;i++)
	{
		for(j=0;j<input_tanh->w_size[i];j++)
		{
			free(input_tanh->W[i][j]);
		}
		free(input_tanh->W[i]);
		
	}
	free(input_tanh->b);
	
	for(i=0;i<input_tanh->hidden_num_layers+2;i++)
	{
		free(input_tanh->out[i]);
		free(input_tanh->in[i]);
	}
	free(input_tanh->out);
	free(input_tanh->in);
	free(input_tanh->w_size);

	
	
	//free output tanh
	for(i=0;i<output_tanh->hidden_num_layers+1;i++)
	{
		for(j=0;j<output_tanh->w_size[i];j++)
		{
			free(output_tanh->W[i][j]);
		}
		free(output_tanh->W[i]);
		
	}
	free(output_tanh->b);
	free(output_tanh->w_size);
	
	
	
	for(i=0;i<output_tanh->hidden_num_layers+2;i++)
	{
		free(output_tanh->out[i]);
		free(output_tanh->in[i]);
	}
	free(output_tanh->out);
	free(output_tanh->in);
	



    //free the full connected layer
    for(i=0;i<final_fully_connected_layer->w_size[0];i++)
    {
		free(final_fully_connected_layer->W[i]);
		
	}
	free(final_fully_connected_layer->W);
	free(final_fully_connected_layer->b);
	free(final_fully_connected_layer->w_size);
	


	//free the objects
	free(forget_sigmoid);
	free(input_sigmoid);
	free(output_sigmoid);
	free(input_tanh);
	free(output_tanh);
	free(final_fully_connected_layer);
	
	
	//free all the intermediate variables
	
	free(differential_bfs->v);
	free(differential_bfs);
	
	
	free(differential_bis->v);
	free(differential_bis);
	
	
	free(differential_bos->v);
	free(differential_bos);
	
	
	free(differential_bit->v);
	free(differential_bit);
	
	
	free(differential_is->v);
	free(differential_is);
	
	free(differential_os->v);
	free(differential_os);
	
	
	free(differential_ot->v);
	free(differential_ot);
	
	
	free(differential_it->v);
	free(differential_it);
	
	free(differential_fscist->v);
	free(differential_fscist);
	
	
	free(fs->v);
	free(fs);
	
	free(is->v);
	free(is);
	
	free(it->v);
	free(it);
	
	free(os->v);
	free(os);
	
	free(ot->v);
	free(ot);
	
	free(ist->v);
	free(ist);
	
	free(ost->v);
	free(os);
	
	free(fsc->v);
	free(fsc);
	
	free(fscist->v);
	free(fscist);
	
	free(combined->v);
	free(combined);
	
	
	free(current_cellstate_output->v);
	free(current_cellstate_output);
	
	free(current_hidden_output->v);
	free(current_hidden_output);
	
	
	free(current_cellstate_differential->v);
	free(current_cellstate_differential);
	
	free(previous_cellstate_output->v);
	free(previous_cellstate_output);
	
	free(previous_hidden_output->v);
	free(previous_hidden_output);
	
	free(incoming_cellstate_differential->v);
	free(incoming_cellstate_differential);
	
	free(incoming_hidden_output_differential->v);
	free(incoming_hidden_output_differential);
	
	
}


//will do the backpropogation of the cell taking as input the succeding cell state differntial and the succeeding hiddenstate differential
//assumes the inputs from the right side are populated
//current target has to be supplied
//target should ideally be a class representation
void cell::cell_backpropogation(vect *t)
{
	
	int i,j;
	
	floater f1,f2;
	//assumes incoming_cellstate_differential and incoming_hidden_output_differntial are populated
	//first find the cross entropy between the taget vector and the current hidden output
	
	cout<<"incoming_hidenstate_differential\n";
	incoming_cellstate_differential->display_vector();
	
	
	cout<<"incoming_cellstate_differential\n";
	incoming_hidden_output_differential->display_vector();
	
	getchar();
	
	
	//get the differentials
	//differential with respect to hidden state
	vect *temp_ost=(vect *)malloc(sizeof(vect));
	temp_ost->v=(floater *)malloc(ost_reduced->v_size*sizeof(floater));
	temp_ost->v_size=ost_reduced->v_size;
	
	//populate 
	for(i=0;i<ost_reduced->v_size;i++)
	{
		temp_ost->v[i]=ost_reduced->v[i]-t->v[i];
	}
	
	
	
	
	//temp_ost is the loss vector to the final fully connected layer for backpropogation
	
	//pass temp_ost to the backprogration of final_fully_connected_layer
	//this will populate the differential_outgoing of the final fully connected layer
	final_fully_connected_layer->backpropogation(temp_ost);
	
	
	
	
	//now add the differential from the succeeding cell to this hidden state differential
	differential_ost=add_vectors(final_fully_connected_layer->differential_outgoing, incoming_hidden_output_differential);	
	
	
	
	
	//both of these vectors are just before the pointwise multiplication
	//populate
	differential_ot=pointwise_product(differential_ost, os);
	differential_os=pointwise_product(differential_ost, ot);
	
	
	

		
	
	
	//do the backpropogation through the output_tanh with differential_ot
	//this will populate outgoing differential
	output_tanh->backpropogation(differential_ot);
	
	
	
	
	//add differential_outgoing to incoming_cellstate_differential to get differential of cell state
	differential_fscist=add_vectors(output_tanh->differential_outgoing, incoming_cellstate_differential);


	      	
	
	//now get differential_it and differntial_is
	differential_it=pointwise_product(differential_fscist, is);
	differential_is=pointwise_product(differential_fscist, it);
	
	//populate the differntial_bfs and differntial_bis
	input_sigmoid->backpropogation(differential_is);
	input_tanh->backpropogation(differential_it);
	
	
	differential_bis=input_sigmoid->differential_outgoing;
	differential_bit=input_tanh->differential_outgoing;
	
	
	
	
	//populate the differential_bos
	output_sigmoid->backpropogation(differential_os);
	differential_bos=output_sigmoid->differential_outgoing;
	
	
	
	
	
	
	//populate current_cellstate_differential
	//this is the outgoing differential from a cell to its preceeding cell
	current_cellstate_differential=pointwise_product(differential_fscist, fs);
	
	//populate differential_fs
	differential_fs=pointwise_product(differential_fscist, previous_cellstate_output);
	
	//populate differnital_bfs
	forget_sigmoid->backpropogation(differential_fs);
	differential_bfs=forget_sigmoid->differential_outgoing;
	
	
	
	
	
	
	//now update the current hiddenstate differential
	vect *temp1=add_vectors(differential_bfs, differential_bis);
	vect *temp2=add_vectors(differential_bit,differential_bos);
	
	
	vect *temp3=add_vectors(temp1, temp2);
	
	
	
	//take only the first O from temp3 and pass it as the current_hidden_output_differential
	current_hidden_output_differential=(vect *)malloc(sizeof(vect ));
	current_hidden_output_differential->v_size=O;
	current_hidden_output_differential->v=(floater *)malloc(O*sizeof(floater));
	
	for(i=0;i<O;i++)
	{
		current_hidden_output_differential->v[i]=temp3->v[i];
	}
	
	
	//free temps
	free(temp1->v);
	free(temp1);
	
	free(temp2->v);
	free(temp2);
	
	free(temp3->v);
	free(temp3);
	
	free(temp_ost->v);
	free(temp_ost);
	
	
	
	
} 


//print all the intermediate vectors in the forward pass
void cell::print_intermediate_vectors_forward_pass()
{
	int i,j;
	cout<<"combined\n";
	combined->display_vector();
	cout<<"fs\n";
	fs->display_vector();
	cout<<"fsc\n";
	fsc->display_vector();
	
	cout<<"is\n";
	is->display_vector();
	cout<<"it\n";
	it->display_vector();
	
	cout<<"ist\n";
	ist->display_vector();
	
	cout<<"fscist\n";
	fscist->display_vector();
	
	
	
	cout<<"os\n";
	os->display_vector();
	cout<<"ot\n";
	ot->display_vector();
	cout<<"ost\n";
	ost->display_vector();
	cout<<"ost_reduced\n";
	ost_reduced->display_vector();	
		
		
	
}

//print all the vectors in backward pass
void cell::print_intermediate_vector_backward_pass()
{
	cout<<"incoming_cellstate_differential\n";
	incoming_cellstate_differential->display_vector();
	cout<<"incoming_hidden_output_differential\n";
	incoming_hidden_output_differential->display_vector();
	cout<<"differential_ost\n";
	differential_ost->display_vector();
	cout<<"differential_os\n";
	differential_os->display_vector();
	cout<<"differential_ot\n";
	differential_ot->display_vector();
	cout<<"differential_fscist\n";
	differential_fscist->display_vector();
	cout<<"differential_it\n";
	differential_it->display_vector();
	cout<<"differential_is\n";
	differential_it->display_vector();
	cout<<"differential_fs\n";
	differential_fs->display_vector();
	cout<<"differential_bos\n";
	differential_bos->display_vector();
	cout<<"differential_bit\n";
	differential_bit->display_vector();
	cout<<"differential_bis\n";
	differential_bis->display_vector();
	cout<<"differential_bfs\n";
	differential_bfs->display_vector();
	cout<<"current_cellstate_output_differential\n";
	current_cellstate_differential->display_vector();
	cout<<"current_hidden_output_differential\n";
	current_hidden_output_differential->display_vector();
	
	
	
	
	
}










//print the output of the cell
void cell::print_outputs()
{
	int i;
	for(i=0;i<O;i++)
	{
		cout<<current_output->v[i]<<" ";
	}
		
}







//this is not strict initialization. just memory allocation
void cell::initialize_parameters()
{
	
	
	int i,j,k;
	
		
	//manually assign
	forget_sigmoid=(sigmoid_layer *)malloc(sizeof(sigmoid_layer));
	forget_sigmoid->hidden_num_layers=NUM_HIDDEN_LAYERS;
	forget_sigmoid->w_size=(int *)malloc((forget_sigmoid->hidden_num_layers+2)*sizeof(int));
	forget_sigmoid->w_size[0]=N+O;
	forget_sigmoid->w_size[1]=H;
	forget_sigmoid->w_size[2]=H;
	forget_sigmoid->w_size[3]=H;
	forget_sigmoid->w_size[4]=M;
	
	forget_sigmoid->W=(floater ***)malloc((forget_sigmoid->hidden_num_layers+1)*sizeof(floater **));
	for(i=0;i<(forget_sigmoid->hidden_num_layers+1);i++)
	{
		forget_sigmoid->W[i]=(floater **)malloc(forget_sigmoid->w_size[i]*sizeof(floater *));
		for(j=0;j<forget_sigmoid->w_size[i];j++)
		{
				forget_sigmoid->W[i][j]=(floater *)malloc(forget_sigmoid->w_size[i+1]*sizeof(floater));
		}
		
	}
	
	forget_sigmoid->b=(floater *)malloc((forget_sigmoid->hidden_num_layers+2)*sizeof(floater));
	
	//first initialize the  out of forget_sigmoid
	forget_sigmoid->out =(floater **)malloc((forget_sigmoid->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<forget_sigmoid->hidden_num_layers+2;i++)
	{
		forget_sigmoid->out[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	//initialize the in of forget_sigmoid
	forget_sigmoid->in =(floater **)malloc((forget_sigmoid->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<forget_sigmoid->hidden_num_layers+2;i++)
	{
		forget_sigmoid->in[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	
	
	
	//for input sigmoid
	//manually assign
	input_sigmoid=(sigmoid_layer *)malloc(sizeof(sigmoid_layer));
	input_sigmoid->hidden_num_layers=NUM_HIDDEN_LAYERS;
	input_sigmoid->w_size=(int *)malloc((input_sigmoid->hidden_num_layers+2)*sizeof(int));
	input_sigmoid->w_size[0]=N+O;
	input_sigmoid->w_size[1]=H;
	input_sigmoid->w_size[2]=H;
	input_sigmoid->w_size[3]=H;
	input_sigmoid->w_size[4]=M;
	
	input_sigmoid->W=(floater ***)malloc((input_sigmoid->hidden_num_layers+1)*sizeof(floater **));
	for(i=0;i<(input_sigmoid->hidden_num_layers+1);i++)
	{
		input_sigmoid->W[i]=(floater **)malloc(input_sigmoid->w_size[i]*sizeof(floater *));
		for(j=0;j<input_sigmoid->w_size[i];j++)
		{
				input_sigmoid->W[i][j]=(floater *)malloc(input_sigmoid->w_size[i+1]*sizeof(floater));
		}
		
	}
	
	input_sigmoid->b=(floater *)malloc((input_sigmoid->hidden_num_layers+2)*sizeof(floater));
	
	//initialize out for input sigmoid
	input_sigmoid->out =(floater **)malloc((input_sigmoid->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<input_sigmoid->hidden_num_layers+2;i++)
	{
		input_sigmoid->out[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	
	
	//initialize in for input sigmoid
	input_sigmoid->in =(floater **)malloc((input_sigmoid->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<input_sigmoid->hidden_num_layers+2;i++)
	{
		input_sigmoid->in[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	
	//output sigmoid
	//manually assign
	output_sigmoid=(sigmoid_layer *)malloc(sizeof(sigmoid_layer));
	output_sigmoid->hidden_num_layers=NUM_HIDDEN_LAYERS;
	output_sigmoid->w_size=(int *)malloc((output_sigmoid->hidden_num_layers+2)*sizeof(int));
	output_sigmoid->w_size[0]=N+O;
	output_sigmoid->w_size[1]=H;
	output_sigmoid->w_size[2]=H;
	output_sigmoid->w_size[3]=H;
	output_sigmoid->w_size[4]=O;
	
	output_sigmoid->W=(floater ***)malloc((output_sigmoid->hidden_num_layers+1)*sizeof(floater **));
	for(i=0;i<(output_sigmoid->hidden_num_layers+1);i++)
	{
		output_sigmoid->W[i]=(floater **)malloc(output_sigmoid->w_size[i]*sizeof(floater *));
		for(j=0;j<output_sigmoid->w_size[i];j++)
		{
				output_sigmoid->W[i][j]=(floater *)malloc(output_sigmoid->w_size[i+1]*sizeof(floater));
		}
		
	}
	
	output_sigmoid->b=(floater *)malloc((output_sigmoid->hidden_num_layers+2)*sizeof(floater));
	
	//initialize out of output sigmoid
	output_sigmoid->out =(floater **)malloc((output_sigmoid->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<output_sigmoid->hidden_num_layers+2;i++)
	{
		output_sigmoid->out[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	//initialize in of output sigmoid
	output_sigmoid->in =(floater **)malloc((output_sigmoid->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<output_sigmoid->hidden_num_layers+2;i++)
	{
		output_sigmoid->in[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	
	//tanh input
	//manually assign
	input_tanh=(tanh_layer *)malloc(sizeof(tanh_layer));
	input_tanh->hidden_num_layers=NUM_HIDDEN_LAYERS;
	input_tanh->w_size=(int *)malloc((input_tanh->hidden_num_layers+2)*sizeof(int));
	input_tanh->w_size[0]=N+O;
	input_tanh->w_size[1]=H;
	input_tanh->w_size[2]=H;
	input_tanh->w_size[3]=H;
	input_tanh->w_size[4]=M;
	
	input_tanh->W=(floater ***)malloc((input_tanh->hidden_num_layers+1)*sizeof(floater **));
	for(i=0;i<(input_tanh->hidden_num_layers+1);i++)
	{
		input_tanh->W[i]=(floater **)malloc(input_tanh->w_size[i]*sizeof(floater *));
		for(j=0;j<input_tanh->w_size[i];j++)
		{
				input_tanh->W[i][j]=(floater *)malloc(input_tanh->w_size[i+1]*sizeof(floater));
		}
		
	}
	
	input_tanh->b=(floater *)malloc((input_tanh->hidden_num_layers+2)*sizeof(floater));
	
	//initialize out of input_tanh
	input_tanh->out =(floater **)malloc((input_tanh->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<input_tanh->hidden_num_layers+2;i++)
	{
		input_tanh->out[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	//initialize in of input_tanh
	input_tanh->in =(floater **)malloc((input_tanh->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<input_tanh->hidden_num_layers+2;i++)
	{
		input_tanh->in[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	//output tanh
	//manually assign
	
	output_tanh=(tanh_layer *)malloc(sizeof(tanh_layer));
	output_tanh->hidden_num_layers=NUM_HIDDEN_LAYERS;
	output_tanh->w_size=(int *)malloc((output_tanh->hidden_num_layers+2)*sizeof(int));
	output_tanh->w_size[0]=M;
	output_tanh->w_size[1]=H;
	output_tanh->w_size[2]=H;
	output_tanh->w_size[3]=H;
	output_tanh->w_size[4]=O;
	
	output_tanh->W=(floater ***)malloc((output_tanh->hidden_num_layers+1)*sizeof(floater **));
	for(i=0;i<(output_tanh->hidden_num_layers+1);i++)
	{
		output_tanh->W[i]=(floater **)malloc(output_tanh->w_size[i]*sizeof(floater *));
		for(j=0;j<output_tanh->w_size[i];j++)
		{
				output_tanh->W[i][j]=(floater *)malloc(output_tanh->w_size[i+1]*sizeof(floater));
		}
		
	}
	
	output_tanh->b=(floater *)malloc((output_tanh->hidden_num_layers+2)*sizeof(floater));
    //initialize out of output_tanh
	output_tanh->out =(floater **)malloc((output_tanh->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<output_tanh->hidden_num_layers+2;i++)
	{
		output_tanh->out[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	//initialize in of output_tanh
	output_tanh->in =(floater **)malloc((output_tanh->hidden_num_layers+2)*sizeof(floater*));
	for(i=0;i<output_tanh->hidden_num_layers+2;i++)
	{
		output_tanh->in[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	
	
	
	//finally allocate memory for final_fully_connected_layer
	final_fully_connected_layer=(fully_connected_layer *)malloc(sizeof(fully_connected_layer));
	final_fully_connected_layer->w_size=(int *)malloc(2*sizeof(int));	//just 2 layers
	
	final_fully_connected_layer->w_size[0]=O;
	final_fully_connected_layer->w_size[1]=global_data_y_dim;
	
	final_fully_connected_layer->b=(floater *)malloc(2*sizeof(floater));
	
	
	
	final_fully_connected_layer->W=(floater **)malloc(O*sizeof(floater *));                        //size Oxglobal_data_y_dim
	for(i=0;i<O;i++)
	{
		final_fully_connected_layer->W[i]=(floater *)malloc(global_data_y_dim*sizeof(floater));
	}
	
	//initialize in
	final_fully_connected_layer->in=(floater **)malloc(2*sizeof(floater *));                   //2 layers
	for(i=0;i<2;i++)
	{
		final_fully_connected_layer->in[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	
	//initialize out
	final_fully_connected_layer->out=(floater **)malloc(2*sizeof(floater *));                   //2 layers
	for(i=0;i<2;i++)
	{
		final_fully_connected_layer->out[i]=(floater *)malloc(MAX_NODE_SIZE*sizeof(floater));
	}
	


	//randomly initialize all the weights and bias of all the fully connected networks
	forget_sigmoid->random_initialize();
	input_sigmoid->random_initialize();
	output_sigmoid->random_initialize();
	input_tanh->random_initialize();
	output_tanh->random_initialize();
    final_fully_connected_layer->random_initialize();
	
		
}




//
//assumes previous_hidden_output and previous_cellstate_output are present
//the current input has to be supplied
void cell::compute_outputs(vect *a)
{
	int i,j;
	//concatenate previous_hidden_output and input
    combined=concat_vector(previous_hidden_output, a);
	
	//forward pass of combined on forget_sigmoid
	 fs=forget_sigmoid->forward_pass(combined);
	
	//forward pass of combined on input_sigmoid
	is=input_sigmoid->forward_pass(combined);
	
	//forward pass of combined on input_tanh
     it=input_tanh->forward_pass(combined);
	
	//pointwise product of previous cellste and ft
	fsc=pointwise_product(fs,previous_cellstate_output);
	
	//pointwise produt
	ist=pointwise_product(is, it);
	
	//get current cell state
	fscist=add_vectors(fsc,ist);
	
	//output sigmoid on combined
	os=output_sigmoid->forward_pass(combined);
	
	//output tanh on fscist
	ot=output_tanh->forward_pass(fscist);
	
	//final pointwise product
	ost=pointwise_product(os, ot);
	
	ost_reduced=final_fully_connected_layer->forward_pass(ost);
	
	
	//update current_hidden_output and current_cellstate_output
	current_hidden_output=ost;
	current_cellstate_output=fscist;
	current_output=current_hidden_output;
	
}


//initialize all the weights in the fully connected layer randomly
void fully_connected_layer::random_initialize()
{
	int i,j;
	for(i=0;i<w_size[0];i++)
	{
		for(j=0;j<w_size[1];j++)
		{
			W[i][j]=(floater) rand()/RAND_MAX ;
		}
		
	}
	b[0]=(floater) rand()/RAND_MAX ;
	b[1]=(floater) rand()/RAND_MAX ;
	
	
	
}




//initialize all the weights randomly
void sigmoid_layer::random_initialize()
{
	int i,j;
	
	
	//initialize w[0]
	for(i=0;i<w_size[0];i++)
	{
		for(j=0;j<w_size[1];j++)
		{
			W[0][i][j]=(floater) rand()/RAND_MAX ;
		}
		
	}
	
	
	//initialize w[1]
	for(i=0;i<w_size[1];i++)
	{
		for(j=0;j<w_size[2];j++)
		{
			W[1][i][j]=(floater) rand()/RAND_MAX ;
		}
		
	}
	
	
	
	//initialize w[2]
	for(i=0;i<w_size[2];i++)
	{
		for(j=0;j<w_size[3];j++)
		{
			W[2][i][j]=(floater) rand()/RAND_MAX ;
		}
	}
	
	
	
	
	
	//initialize w[3]
	for(i=0;i<w_size[3];i++)
	{
		for(j=0;j<w_size[4];j++)
		{
			W[3][i][j]=(floater) rand()/RAND_MAX ;
		}
	}
	
	//imitalize b
	b[0]=(floater) rand()/RAND_MAX;
	b[1]=(floater) rand()/RAND_MAX;
	b[2]=(floater) rand()/RAND_MAX;
	b[3]=(floater) rand()/RAND_MAX;
	b[4]=(floater) rand()/RAND_MAX;
	
}






//initialize all the weights randomly
void tanh_layer::random_initialize()
{
	int i,j;
	
	
	//initialize w[0]
	for(i=0;i<w_size[0];i++)
	{
		for(j=0;j<w_size[1];j++)
		{
			W[0][i][j]=(floater) rand()/RAND_MAX ;
		}
		
	}
	
	
	//initialize w[1]
	for(i=0;i<w_size[1];i++)
	{
		for(j=0;j<w_size[2];j++)
		{
			W[1][i][j]=(floater) rand()/RAND_MAX ;
		}
		
	}
	
	
	
	//initialize w[2]
	for(i=0;i<w_size[2];i++)
	{
		for(j=0;j<w_size[3];j++)
		{
			W[2][i][j]=(floater) rand()/RAND_MAX ;
		}
	}
	
	
	
	
	
	//initialize w[3]
	for(i=0;i<w_size[3];i++)
	{
		for(j=0;j<w_size[4];j++)
		{
			W[3][i][j]=(floater) rand()/RAND_MAX ;
		}
	}
	
	//imitalize b
	b[0]=(floater) rand()/RAND_MAX;
	b[1]=(floater) rand()/RAND_MAX;
	b[2]=(floater) rand()/RAND_MAX;
	b[3]=(floater) rand()/RAND_MAX;
	b[4]=(floater) rand()/RAND_MAX;
	
}






//forward pass of fully connected layer
//final layer is a softmmax layer
//output of the softmax is the output of the forward_pass
vect *fully_connected_layer::forward_pass(vect *input)
{
	
	//check if the input vector size corresponds to the networks input size
	if(input->v_size!=w_size[0])
	{
		cout <<"This vector cannot go through the fully connected layer\n";
		exit(0); 
	}
	
	int i,j;
	floater f1, f2;
	//do each layer separately
	
	//first apply the input to the in
	for(i=0;i<input->v_size;i++)
	{
		in[0][i]=input->v[i];
		out[0][i]=in[0][i];
	}
	
	
	//popualte the second layer of ins
	for(j=0;j<w_size[1];j++)
	{
		f1=0;
		for(i=0;i<w_size[0];i++)
		{
			
			f1=f1+out[0][i]*W[i][j];
		}
		in[1][j]=f1;
	}
	
	//take a soft max of in[1] and return the vector
	//first convert the in into a vector
	vect *temp_in=(vect *)malloc(sizeof(vect));
	temp_in->v_size=w_size[1];
	temp_in->v=(floater *)malloc(w_size[1]*sizeof(floater));
	for(i=0;i<w_size[1];i++)
	{
		temp_in->v[i]=in[1][i];
	}
	
	
	vect *reduced_vector=vect_to_softmax(temp_in);
	
	
	//free temp_in
	free(temp_in->v);
	free(temp_in);
	
	return reduced_vector;
	
}










//forward pass of a vector through the full sigmoid layer
vect * sigmoid_layer::forward_pass(vect *a)
{
	int i,j;
	floater f1, f2;
	
	
	//first check whether the size of the vector is the same as the size of the first layer 
	if(w_size[0]!=a->v_size)
	{
		cout<<"Forward pass: Size of the input vector to the sigmoid layer is wrong\n";
		cout<<w_size[0]<<" "<<a->v_size<<"\n";
		exit(0);
	}
	
	//populate the out of the first layer as the input vector
	for(i=0;i<w_size[0];i++)
	{
		in[0][i]=a->v[i];
		out[0][i]=in[0][i];
	}
	
	//populate the out of the second layer
	for(i=0;i<w_size[1];i++)
	{
		f1=0;	
		for(j=0;j<w_size[0];j++)
		{
			f1=f1+out[0][j]*W[0][j][i];
		}
		//add bias
		f1=f1+b[1];
		//updat the in[1]
		in[1][i]=f1;		
		//now update out[1]
		out[1][i]=sigmoid(in[1][i]);
	}
	
	//populate out of third layer
	for(i=0;i<w_size[2];i++)
	{
		f1=0;	
		for(j=0;j<w_size[1];j++)
		{
			f1=f1+out[1][j]*W[1][j][i];
		}
		
		//add bias
		f1=f1+b[2];
		
		in[2][i]=f1;
		//now update out[1]
		out[2][i]=sigmoid(in[2][i]);
	}
	
	//populate out of fourth layer
	for(i=0;i<w_size[3];i++)
	{
		f1=0;	
		for(j=0;j<w_size[2];j++)
		{
			f1=f1+out[2][j]*W[2][j][i];
		}
		//add bias
		f1=f1+b[3];
		
		in[3][i]=f1;
		//now update out[1]
		out[3][i]=sigmoid(in[3][i]);
	}
	
	
	
	
	//populate out of fifth layer
	for(i=0;i<w_size[4];i++)
	{
		f1=0;	
		for(j=0;j<w_size[3];j++)
		{
			f1=f1+out[3][j]*W[3][j][i];
		}
		//add bias
		f1=f1+b[4];
		in[4][i]=f1;
		//now update out[1]
		out[4][i]=sigmoid(in[4][i]);
	}
	
	
	//get the out[4] into a vect and return
	
	vect *c=(vect *)malloc(sizeof(vect));
	
	//new vector has the output size of last sigmoid layer
	c->v_size=w_size[4];
	c->v=(floater *)malloc(c->v_size*sizeof(floater));
	for(i=0;i<c->v_size;i++)
	{
		c->v[i]=out[4][i];
	}
	return c;
}







//forward pass of a vector through the full tanh layer
vect *tanh_layer::forward_pass(vect *a)
{
	int i,j;
	floater f1, f2;
	
	
	//first check whether the size of the vector is the same as the size of the first layer 
	if(w_size[0]!=a->v_size)
	{
		cout<<"Forward pass error: Size of the input vector to the tanh layer is wrong\n";
		exit(0);
	}
	
	//populate the out of the first layer as the input vector
	for(i=0;i<w_size[0];i++)
	{
		in[0][i]=a->v[i];
		out[0][i]=in[0][i];
	}
	
	//populate the out of the second layer
	for(i=0;i<w_size[1];i++)
	{
		f1=0;	
		for(j=0;j<w_size[0];j++)
		{
			f1=f1+out[0][j]*W[0][j][i];
		}
		f1=f1+b[1];
		in[1][i]=f1;
		//now update out[1]
		out[1][i]=tanh(in[1][i]);
	}
	
	//populate out of third layer
	for(i=0;i<w_size[2];i++)
	{
		f1=0;	
		for(j=0;j<w_size[1];j++)
		{
			f1=f1+out[1][j]*W[1][j][i];
		}
		f1=f1+b[2];
		//now update out[1]
		
		in[2][i]=f1;
		out[2][i]=tanh(in[2][i]);
	}
	
	//populate out of fourth layer
	for(i=0;i<w_size[3];i++)
	{
		f1=0;	
		for(j=0;j<w_size[2];j++)
		{
			f1=f1+out[2][j]*W[2][j][i];
		}
		f1=f1+b[3];
		//now update out[1]
		
		in[3][i]=f1;
		out[3][i]=tanh(in[3][i]);
	}
	
	
	
	
	//populate out of fifth layer
	for(i=0;i<w_size[4];i++)
	{
		f1=0;	
		for(j=0;j<w_size[3];j++)
		{
			f1=f1+out[3][j]*W[3][j][i];
		}
		f1=f1+b[4];
		//now update out[1]
		in[4][i]=f1;
		out[4][i]=tanh(in[4][i]);
	}
	
	
	//get the out[4] into a vect and return
	
	vect *c=(vect *)malloc(sizeof(vect));
	
	//new vector has the output size of last sigmoid layer
	c->v_size=w_size[4];
	c->v=(floater *)malloc(c->v_size*sizeof(floater));
	for(i=0;i<c->v_size;i++)
	{
		c->v[i]=out[4][i];
	}
	return c;
}




//print the weights of the sigmoid layer
void sigmoid_layer::print_weights()
{
	
	int i,j;
	
	cout<<"layer:0-1\n";
	
	for(i=0;i<w_size[0];i++)
	{
		for(j=0;j<w_size[1];j++)
		{
			cout<<i<<":"<<j<<":"<<W[0][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	cout<<"layer:1-2\n";
	for(i=0;i<w_size[1];i++)
	{
		for(j=0;j<w_size[2];j++)
		{
			cout<<i<<":"<<j<<":"<<W[1][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	
	cout<<"layer:2-3\n";
	for(i=0;i<w_size[2];i++)
	{
		for(j=0;j<w_size[3];j++)
		{
			cout<<i<<":"<<j<<":"<<W[2][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	
	cout<<"layer:3-4\n";
	for(i=0;i<w_size[3];i++)
	{
		for(j=0;j<w_size[4];j++)
		{
			cout<<i<<":"<<j<<":"<<W[3][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	
}



//print the weights of the tanh layer
void tanh_layer::print_weights()
{
	
	int i,j;
	
	cout<<"layer:0-1\n";
	
	for(i=0;i<w_size[0];i++)
	{
		for(j=0;j<w_size[1];j++)
		{
			cout<<i<<":"<<j<<":"<<W[0][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	cout<<"layer:1-2\n";
	for(i=0;i<w_size[1];i++)
	{
		for(j=0;j<w_size[2];j++)
		{
			cout<<i<<":"<<j<<":"<<W[1][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	
	cout<<"layer:2-3\n";
	for(i=0;i<w_size[2];i++)
	{
		for(j=0;j<w_size[3];j++)
		{
			cout<<i<<":"<<j<<":"<<W[2][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	
	cout<<"layer:3-4\n";
	for(i=0;i<w_size[3];i++)
	{
		for(j=0;j<w_size[4];j++)
		{
			cout<<i<<":"<<j<<":"<<W[3][i][j]<<" ";
		}
		cout<<"\n";
	}
	
	
}





//print the bias of sigmoid layer
void sigmoid_layer::print_bias()
{
	int i;
	
	cout<<"bias\n";
	for(i=0;i<hidden_num_layers+2;i++)
	{
		cout<<b[i]<<" ";
	}
	cout<<"\n";
	
}






//print the bias of sigmoid layer
void tanh_layer::print_bias()
{
	int i;
	
	cout<<"bias\n";
	for(i=0;i<hidden_num_layers+2;i++)
	{
		cout<<b[i]<<" ";
	}
	cout<<"\n";
	
}








//print the outs of sigmoid layer
void sigmoid_layer::print_outs()
{
	int i,j;
	
	cout<<"out\n";
	for(i=0;i<hidden_num_layers+2;i++)
	{
		cout<<"out layer:"<<i<<"\n";
		for(j=0;j<w_size[i];j++)
		{
			
			cout<<j<<":"<<out[i][j]<<" ";
			
		}
		cout<<"\n";
		
	}
	
	
	
}




//print the ins of sigmoid layer
void sigmoid_layer::print_ins()
{
	int i,j;
	
	cout<<"in\n";
	for(i=0;i<hidden_num_layers+2;i++)
	{
		cout<<"in layer:"<<i<<"\n";
		for(j=0;j<w_size[i];j++)
		{
			
			cout<<j<<":"<<in[i][j]<<" ";
			
		}
		cout<<"\n";
		
	}
	
}






//print the outs of tanh layer
void tanh_layer::print_outs()
{
	int i,j;
	
	cout<<"out\n";
	for(i=0;i<hidden_num_layers+2;i++)
	{
		cout<<"layer:"<<i<<"\n";
		for(j=0;j<w_size[i];j++)
		{
			
			cout<<out[i][j]<<" ";
			
		}
		cout<<"\n";
		
	}
}









//print the ins of tanh layer
void tanh_layer::print_ins()
{
	int i,j;
	
	cout<<"in\n";
	for(i=0;i<hidden_num_layers+2;i++)
	{
		cout<<"layer:"<<i<<"\n";
		for(j=0;j<w_size[i];j++)
		{
			
			cout<<in[i][j]<<" ";
			
		}
		cout<<"\n";
		
	}

}



//this will take a loss vector which is the derivative of cross entropy with the softmax 
//ak-tk is the loss vector content
//will populate the outgoing_differential
void fully_connected_layer::backpropogation(vect *loss_vector)
{
	 int i,j;
	 floater f1;
	 
	 //update thwe weights directly
	 for(i=0;i<w_size[0];i++)
	 {
		 for(j=0;j<w_size[1];j++)
		 {
			 W[i][j]=W[i][j]+LEARNING_RATE*(out[0][i]*loss_vector->v[j]);
		 }
	 }
	
	
	//just dont update bias for now
	
	//populate the outgoing diferential
	
	differential_outgoing=(vect *)malloc(sizeof(vect));
	differential_outgoing->v_size=w_size[0];
	differential_outgoing->v=(floater *)malloc(w_size[0]*sizeof(floater));
	
	for(i=0;i<w_size[0];i++)
	{
		f1=0;
		for(j=0;j<w_size[1];j++)
		{
			f1=f1+W[i][j]*loss_vector->v[j];
		}
		differential_outgoing->v[i]=f1;
		
	}
	
	
	
	
}




//do the backpropogation and update all weights and bias and populate the differential_outgoing vector
void tanh_layer::backpropogation(vect *loss_vector)
{
	
	
	
	int i,j,k,g,h;
	floater f1, f2,f3,f4,f5,f6,f7,f8,f9,f10;
	floater c1, c2, c3;
	
	//first update the last layer
	//update j-k
	for(j=0;j<w_size[3];j++)
	{
		for(k=0;k<w_size[4];k++)
		{
			f1=loss_vector->v[k]*derivative_tanh(in[4][k])*out[3][j];
			c1=loss_vector->v[k]*derivative_tanh(in[4][k]);
			
			W[3][j][k]=W[3][j][k]+LEARNING_RATE*f1;
			b[4]=b[4]+LEARNING_RATE*c1;
		}
	}
	
		
	//update  i-j
	f1=0;
	for(k=0;k<w_size[4];k++)
	{
			f1=f1+derivative_tanh(in[4][k])*loss_vector->v[k];
		
	}//end of k	
			
	
	for(i=0;i<w_size[2];i++)
	{
		for(j=0;j<w_size[3];j++)
		{
	
			
			
			f2=0;
			for(k=0;k<w_size[4];k++)
			{
				f2=f2+W[3][j][k];
		
			}//end of k	
			
			f3=	derivative_tanh(in[3][j]);
			f4=out[2][i];
			f5=f1*f2*f3*f4;
			c1=f1*f2*f3;
			W[2][i][j]=W[2][i][j]+LEARNING_RATE*f5;
			b[3]=b[3]+LEARNING_RATE*c1;
			
		}//end of j
	}//end of i
	
	
	
	//update h-i
	
	f1=0;
	for(k=0;k<w_size[4];k++)
	{
			f1=f1+derivative_tanh(in[4][k])*loss_vector->v[k];			
	}
	
	f3=0;
	for(j=0;j<w_size[3];j++)
	{
		f3=f3+derivative_tanh(in[3][j]);
	}
	f2=0;
	for(j=0;j<w_size[3];j++)
	{
		for(k=0;k<w_size[4];k++)
		{
				f2=f2+W[3][j][k];
		}
	}
	
	for(h=0;h<w_size[1];h++)
	{
		for(i=0;i<w_size[2];i++)
		{
				f4=0;
				for(j=0;j<w_size[3];j++)
				{
					f4=f4+W[2][i][j];
				}
				
			
				f5=derivative_tanh(in[2][i]);
				f6=out[1][h];
				f7=f1*f2*f3*f4*f5*f6;
				c1=f1*f2*f3*f4*f5;
				
			    W[1][h][i]=W[1][h][i]+LEARNING_RATE*f7;
				b[2]=b[2]+LEARNING_RATE*c1;
			
		}//end of i
				
	}//end of h
	
	
	
	
	//update g-h
	f1=0;
	for(k=0;k<w_size[4];k++)
	{
		f1=f1+derivative_tanh(in[4][k])*loss_vector->v[k];			
	}
	
	
	f3=0;
	for(j=0;j<w_size[3];j++)
	{
			f3=f3+derivative_tanh(in[3][j]);
	}
	
	f2=0;
	for(j=0;j<w_size[3];j++)
	{
			for(k=0;k<w_size[4];k++)
			{
					f2=f2+W[3][j][k];
			}
	
	}
	
	f4=0;
	for(i=0;i<w_size[2];i++)
	{
			for(j=0;j<w_size[3];j++)
			{
					f4=f4+W[2][i][j];
			}
		
	}
	
	f5=0;
	for(i=0;i<w_size[2];i++)
	{
			f5=f5+derivative_tanh(in[2][i]);
	}
		
	for(g=0;g<w_size[0];g++)
	{
		for(h=0;h<w_size[1];h++)
		{
			
				
			f6=0;
			for(i=0;i<w_size[2];i++)
			{
				f6=f6+W[1][h][i];
			}
	
			f7=derivative_tanh(in[1][h]);
			f8=out[0][g];
			
			f9=f1*f2*f3*f4*f5*f6*f7*f8;
			c1=f1*f2*f3*f4*f5*f6*f7;
			
			W[0][g][h]=W[0][g][h]+LEARNING_RATE*f9;
			b[1]=b[1]+LEARNING_RATE*c1;
		}//end of h
	
	}//end of g
	
	
	
	
	
	//differential w.r.t input
	
		
	f6=0;
	for(h=0;h<w_size[1];h++)
	{
		for(i=0;i<w_size[2];i++)
		{
			f6=f6+W[1][h][i];
			
		}
		
	}
	
	
	
	
	
	f7=0;
	for(h=0;h<w_size[1];h++)
	{
		f7=f7+derivative_tanh(in[1][h]);
	}
	
	
	//populate the differential_outgoing vector
	differential_outgoing=(vect *)malloc(sizeof(vect ));
	differential_outgoing->v_size=w_size[0];
	differential_outgoing->v=(floater *)malloc(w_size[0]*sizeof(floater));
	
	for(g=0;g<w_size[0];g++)
	{
		f8=0;
		for(h=0;h<w_size[1];h++)
		{
			f8=f8+W[0][g][h];
		}
		f9=1;   //differential of input=output is just 1
		f10=f1*f2*f3*f4*f5*f6*f7*f8*f9;
		
		differential_outgoing->v[g]=f10;
		
	}
	
	
	
	
	
	
}//end of fn







//do the backpropogation and update all weights and bias and populate the differential_outgoing vector
void sigmoid_layer::backpropogation(vect *loss_vector)
{
	int i,j,k,g,h;
	floater f1, f2,f3,f4,f5,f6,f7,f8,f9,f10;
	floater c1, c2, c3;
	
	//first update the last layer
	//update j-k
	for(j=0;j<w_size[3];j++)
	{
		for(k=0;k<w_size[4];k++)
		{
			f1=loss_vector->v[k]*derivative_sigmoid(in[4][k])*out[3][j];
			c1=loss_vector->v[k]*derivative_sigmoid(in[4][k]);
			
			W[3][j][k]=W[3][j][k]+LEARNING_RATE*f1;
			b[4]=b[4]+LEARNING_RATE*c1;
		}
	}
	
	
	
	
	//update  i-j
	f1=0;
	for(k=0;k<w_size[4];k++)
	{
			f1=f1+derivative_sigmoid(in[4][k])*loss_vector->v[k];
		
	}//end of k	
			
	
	for(i=0;i<w_size[2];i++)
	{
		for(j=0;j<w_size[3];j++)
		{
	
			
			
			f2=0;
			for(k=0;k<w_size[4];k++)
			{
				f2=f2+W[3][j][k];
		
			}//end of k	
			
			f3=	derivative_sigmoid(in[3][j]);
			f4=out[2][i];
			f5=f1*f2*f3*f4;
			c1=f1*f2*f3;
			W[2][i][j]=W[2][i][j]+LEARNING_RATE*f5;
			b[3]=b[3]+LEARNING_RATE*c1;
			
		}//end of j
	}//end of i
	
	
	
	//update h-i
	
	f1=0;
	for(k=0;k<w_size[4];k++)
	{
			f1=f1+derivative_sigmoid(in[4][k])*loss_vector->v[k];			
	}
	
	f3=0;
	for(j=0;j<w_size[3];j++)
	{
		f3=f3+derivative_sigmoid(in[3][j]);
	}
	f2=0;
	for(j=0;j<w_size[3];j++)
	{
		for(k=0;k<w_size[4];k++)
		{
				f2=f2+W[3][j][k];
		}
	}
	
	for(h=0;h<w_size[1];h++)
	{
		for(i=0;i<w_size[2];i++)
		{
				f4=0;
				for(j=0;j<w_size[3];j++)
				{
					f4=f4+W[2][i][j];
				}
				
			
				f5=derivative_sigmoid(in[2][i]);
				f6=out[1][h];
				f7=f1*f2*f3*f4*f5*f6;
				c1=f1*f2*f3*f4*f5;
				
			    W[1][h][i]=W[1][h][i]+LEARNING_RATE*f7;
				b[2]=b[2]+LEARNING_RATE*c1;
			
		}//end of i
				
	}//end of h
	
		
	
	//update g-h
	f1=0;
	for(k=0;k<w_size[4];k++)
	{
		f1=f1+derivative_sigmoid(in[4][k])*loss_vector->v[k];			
	}
	
	
	f3=0;
	for(j=0;j<w_size[3];j++)
	{
			f3=f3+derivative_sigmoid(in[3][j]);
	}
	
	f2=0;
	for(j=0;j<w_size[3];j++)
	{
			for(k=0;k<w_size[4];k++)
			{
					f2=f2+W[3][j][k];
			}
	
	}
	
	f4=0;
	for(i=0;i<w_size[2];i++)
	{
			for(j=0;j<w_size[3];j++)
			{
					f4=f4+W[2][i][j];
			}
		
	}
	
	f5=0;
	for(i=0;i<w_size[2];i++)
	{
			f5=f5+derivative_sigmoid(in[2][i]);
	}
		
	for(g=0;g<w_size[0];g++)
	{
		for(h=0;h<w_size[1];h++)
		{
			
				
			f6=0;
			for(i=0;i<w_size[2];i++)
			{
				f6=f6+W[1][h][i];
			}
	
			f7=derivative_sigmoid(in[1][h]);
			f8=out[0][g];
			
			f9=f1*f2*f3*f4*f5*f6*f7*f8;
			c1=f1*f2*f3*f4*f5*f6*f7;
			
			W[0][g][h]=W[0][g][h]+LEARNING_RATE*f9;
			b[1]=b[1]+LEARNING_RATE*c1;
		}//end of h
	
	}//end of g
	
	
	
	//differential w.r.t input
	f1=0;
	for(k=0;k<w_size[4];k++)
	{
		f1=f1+derivative_sigmoid(in[4][k])*loss_vector->v[k];			
	}
	
	
	f3=0;
	for(j=0;j<w_size[3];j++)
	{
			f3=f3+derivative_sigmoid(in[3][j]);
	}
	
	f2=0;
	for(j=0;j<w_size[3];j++)
	{
			for(k=0;k<w_size[4];k++)
			{
					f2=f2+W[3][j][k];
			}
	
	}
	
	f4=0;
	for(i=0;i<w_size[2];i++)
	{
			for(j=0;j<w_size[3];j++)
			{
					f4=f4+W[2][i][j];
			}
		
	}
	
	f5=0;
	for(i=0;i<w_size[2];i++)
	{
			f5=f5+derivative_sigmoid(in[2][i]);
	}
	
	f6=0;
	for(h=0;h<w_size[1];h++)
	{
		for(i=0;i<w_size[2];i++)
		{
			f6=f6+W[1][h][i];
			
		}
		
	}
	
	f7=0;
	for(h=0;h<w_size[1];h++)
	{
		f7=f7+derivative_sigmoid(in[1][h]);
	}
	
	//populate the differential_outgoing vector
	differential_outgoing=(vect *)malloc(sizeof(vect ));
	differential_outgoing->v_size=w_size[0];
	differential_outgoing->v=(floater *)malloc(w_size[0]*sizeof(floater));
	
	for(g=0;g<w_size[0];g++)
	{
		f8=0;
		for(h=0;h<w_size[1];h++)
		{
			f8=f8+W[0][g][h];
		}
		f9=1;   //differential of input=output is just 1
		f10=f1*f2*f3*f4*f5*f6*f7*f8*f9;
		
		differential_outgoing->v[g]=f10;
		
	}
	
	
	
	
}//end of fn








//returns a vector with point wise product
vect *pointwise_product(vect *a, vect *b)
{
	if(a->v_size!=b->v_size)
	{
		cout<<"Differnce in size of vectors in point wise multiplication\n";
		exit(0);
	}
	
	int i;
	vect *c=(vect *)malloc(sizeof(vect));;
	c->v=(floater *)malloc(a->v_size*sizeof(floater));
	for(i=0;i<a->v_size;i++)
	{
		c->v[i]=a->v[i]*b->v[i];
	}
	c->v_size=a->v_size;
	return c;
	
}


//just print the  vector
void vect::display_vector()
{
	int i;
	for(i=0;i<v_size;i++)
	{
			cout<<v[i]<<" ";
	}
	cout<<"\n";
	
	
	
}


//concatenates 2 vect and returns the output
vect *concat_vector(vect *a, vect *b)
{
	int new_size=a->v_size+b->v_size;
	vect *c=(vect *)malloc(sizeof(vect));
	int i,j,k;
	
	c->v=(floater *)malloc(new_size*sizeof(floater));
	k=0;
	for(i=0;i<a->v_size;i++)
	{
		c->v[k++]=a->v[i];
	}
	
	for(i=0;i<b->v_size;i++)
	{
		c->v[k++]=b->v[i];
	}
	c->v_size=k;
	return c;
}



//assumes vect are of same length. otherwise kill the program
vect *add_vectors(vect *a, vect *b)
{
	int i;
	//create a new vect and return
	if(a->v_size!=b->v_size)
	{
		cout<<"vect addition of unequal length "<<a->v_size<<" "<<b->v_size<<"\n";
		exit(0);
	}
	
	vect *c=(vect *)malloc(sizeof(vect));
	c->v=(floater *)malloc(a->v_size*sizeof(floater));
	
	for(i=0;i<a->v_size;i++)
	{
		c->v[i]=a->v[i]+b->v[i];
	}
	c->v_size=a->v_size;
	return c;
}




//assumes vect are of same length. otherwise kill the program
//c=a-b;
vect *sub_vectors(vect *a, vect *b)
{
	int i;
	//create a new vect and return
	if(a->v_size!=b->v_size)
	{
		cout<<"vect addition of unequal length\n";
		exit(0);
	}
	
	vect *c=(vect *)malloc(sizeof(vect));
	c->v=(floater *)malloc(a->v_size*sizeof(floater));
	
	for(i=0;i<a->v_size;i++)
	{
		c->v[i]=a->v[i]-b->v[i];
	}
	c->v_size=a->v_size;
	return c;
}







//sigmoid function
floater sigmoid(floater in)
{	
	floater out;
	out=1/(1+exp(-1*in));
	return out;
	
	
}










//main routine
int main(int argc, char *argv[])
{
	
		int i,j;
		
		srand( (unsigned)time( NULL ) );
	
		if(argc!=5)
		{
			cout<<"./lstm data time_instances x_dim y_dim\n";
			exit(0);
		}
		
		
		//set the global parameters
		global_data_x_dim=atoi(argv[3]);
		
		//set N
		N=global_data_x_dim;
		global_data_y_dim=atoi(argv[4]);
	
		global_data_time_instances=atoi(argv[2]);
		
		
		//read the data
		read_data(argv[1], global_data_time_instances,global_data_x_dim, global_data_y_dim);
		
		
		vect *vect_x_array, *vect_y_array;
		
	
	    lstm *ll=(lstm *)malloc(sizeof(lstm));
	    
	    //create the cell_array with global_data_time_instances
	    //this just initialize all the cells to 0
	    ll->create_cell_array(global_data_time_instances);
	    
	    
	    //never print the outputs here.
	    //outputs haven't been created here.
	    //call a forward pass with a <x,y> time series before printing the lstm output
	
		
		//take each data point from the global array convert it into vector array and call the forward_pass
		for(i=0;i<global_data_num_rows;i++)
		{
			//convert each row into aan array of vectors. a full <x,y> time series
			
			//get the x tiem series
			vect_x_array=create_vector_array(i, true);   			//true will take the x_data. false will take y_data
			//vect_array->display_vector();
			
			//get the corresponding y tiem series
			vect_y_array=create_vector_array(i, false); 
			
			//call the forward pass of lstm with an array of vectors
			
			//do the forward pass
			ll->forward_pass(vect_x_array);								// it is a this time the outputs of each cell is created
		
			cout<<"full cell forward pass over\n";
					
						
			//do the backward pass
			ll->backward_pass(vect_y_array);
			
			getchar();
			//deallocate vect_x_array
			destroy_vector_array(vect_x_array);
			destroy_vector_array(vect_y_array);
			
			
			
		}
	
		
		
		
		//print_data();
		
		//deallocate the global data memory
		for(i=0;i<global_data_num_rows;i++)
		{
			for(j=0;j<global_data_time_instances;j++)
			{
				free(global_data_x[i][j]);
				free(global_data_y[i][j]);
			}
			free(global_data_x[i]);
			free(global_data_y[i]);
			
		}
		free(global_data_x);
		free(global_data_y);
		
		
		
		
		//free cell array
		ll->destroy_cell_array();
		
		//free(lstm object)
		free(ll);
		
		cout<<"over\n";
		
}


//destroy the array used for lstm foward pass and lstm backward pass
//assume array is of size global_data_instance
void destroy_vector_array(vect *vect_array)
{
	int i;
	for(i=0;i<global_data_time_instances;i++)
	{
		free(vect_array[i].v);
		
		
	}
	free(vect_array);	
	
	
}



//read the data into    global variables
void read_data(char *fn, int t, int x_dim, int y_dim)
{
	
	
	int i,j;
	int line_num=0;
	
	char str[MAX_DATA_LINE_WIDTH];
	global_data_x=(floater ***)malloc(MAX_DATA_ROWS*sizeof(floater **));
	global_data_y=(floater ***)malloc(MAX_DATA_ROWS*sizeof(floater **));
	
	global_data_num_rows=0;
	
	
	
	FILE *fp=fopen(fn,"r");
	if(fp==NULL)
	{
		cout<<"The data file couldn't be opened\n";
		exit(0);
	}
	
	
	
	
	
	while(1)
	{
		//read a line and parse it into x and y
		fgets(str, MAX_DATA_LINE_WIDTH, fp);
		
		
		//allocate the memory for a row
		global_data_x[global_data_num_rows]=(floater **)malloc(global_data_time_instances*sizeof(floater*));
		global_data_y[global_data_num_rows]=(floater **)malloc(global_data_time_instances*sizeof(floater*));
		
		for(i=0;i<global_data_time_instances;i++)
		{
			global_data_x[global_data_num_rows][i]=(floater *)malloc(global_data_x_dim*sizeof(floater));
			global_data_y[global_data_num_rows][i]=(floater *)malloc(global_data_y_dim*sizeof(floater));
		}
		
		
		//parse the string
		parse_line(str, line_num);
		
			
		if(feof(fp))break;
		line_num++;
		global_data_num_rows=global_data_num_rows+1;
		
		
	}
	fclose(fp);
	
	
	
}


//str contains an instance of a time series
void parse_line(char str[], int line_num)
{
	
	//cout<<str<<"\n";
	//getchar();
	
	
	int i,j;
	int t=0;
	int l=strlen(str);
	floater f1;
	
	
	
	bool if_x;
	int start_loc, end_loc;
	
	i=0;
	while(1)
	{
	
		
		while(str[i]=='[' || str[i]==']'|| str[i]==',')
		{
			i++;
		}
		// this is the start location of number
		start_loc=i;
	
		//get the end loc of the x
		while(str[i]!=']')
		{
			i++;
		}
	
		end_loc=i-1;
	
	
		//extract between  start and end for x
		if_x=true;
	
		//call for x
		extract(str, start_loc, end_loc, if_x, t);
	
	
	
		//from end location now get the y start
		i=end_loc;
		while(str[i]!='[')
		{
			i++;
		}
		start_loc=i+1;
		//got the start location of the y.
		//now get the end location of y
	
		i=start_loc;
		while(str[i]!=']')
		{
			i++;
		}
		end_loc=i-1;
		if_x=false;
	
		//call for y
		extract(str, start_loc, end_loc, if_x, t);
		t++;
		if(t>=global_data_time_instances)break;
	}
	
}



//get the string and get the start and end location of data and extract.
//c an be used to extract x and y data
void extract(char str[], int start_loc, int end_loc, bool if_x, int t)
{
	
	int i,j;
	
	
	/*
	for(i=start_loc;i<=end_loc;i++)
	{
		cout<<str[i];
	}
	cout<<"\n";
	getchar();
	*/
	
	int k=0;
	int col=0;
	
	char str2[MAX_VAL_WIDTH];
	
	i=start_loc;
	while(1)
	{
		while(str[i]!=',' && i<=end_loc)
		{
			str2[k]=str[i];
			i++;
			k++;
		}
		str2[k]='\0';
		if(if_x==true)
		{
			global_data_x[global_data_num_rows][t][col]=atof(str2);
			//cout<<global_data_x[global_data_num_rows][col]<<"\n";
			//getchar();
		}
		else
		{
			global_data_y[global_data_num_rows][t][col]=atof(str2);
			//cout<<global_data_y[global_data_num_rows][col]<<"\n";
		//	getchar();
		}
		col++;
		k=0;
		i=i+1;
		if(i>end_loc)break;
	}
	
	
	
	
}

//print the data set
void print_data()
{
	
	int i, j, k,l;
	
	for(i=0;i<global_data_num_rows;i++)
	{
		
		for(j=0;j<global_data_time_instances;j++)
		{
			for(k=0;k<global_data_x_dim;k++)
			{
				cout<<global_data_x[i][j][k]<<" ";
			}
			for(k=0;k<global_data_y_dim;k++)
			{
				cout<<global_data_y[i][j][k]<<" ";
			}
		}
		
		cout<<"\n";
	}
	
}



//this will return a vector array either of x or y from the global data array
//vector array contains global_data_time_instance number of vectors, either x or y
vect *create_vector_array(int index, bool is_X)
{
	int i,j;
	
	//size of vector array is the size of time instance
	vect *vect_array=(vect *)malloc(global_data_time_instances*sizeof(vect));
	
	
	//allocate space for individual vects
	for(i=0;i<global_data_time_instances;i++)
	{
		if(is_X==true)
		{
			vect_array[i].v=(floater *)malloc(global_data_x_dim*sizeof(floater));
			vect_array[i].v_size=global_data_x_dim;
		}
		else
		{
			vect_array[i].v=(floater *)malloc(global_data_y_dim*sizeof(floater));
			vect_array[i].v_size=global_data_y_dim;
		}
	}
	
	
	//copy the data
	for(i=0;i<global_data_time_instances;i++)
	{
		if(is_X==true)
		{
			for(j=0;j<global_data_x_dim;j++)
			{
				vect_array[i].v[j]=global_data_x[index][i][j];
			}
		}
		
		
		else
		{
			for(j=0;j<global_data_x_dim;j++)
			{
				vect_array[i].v[j]=global_data_y[index][i][j];
			}
		}
		
		
	}	

	return vect_array;
}





//return the softmax function
floater softmax(vect *a, int index)
{
	
	int i;
	floater f1=0;
	floater f2;
	
	for(i=0;i<a->v_size;i++)
	{
		f1=f1+exp(a->v[i]);
	}
	f2=exp(a->v[index])/f1;
	return f2;
	
	
	
}



//input a vector convert to softmax vector and return
vect *vect_to_softmax(vect *a)
{
	
	
	int i,j;
	vect *softmax_vect =(vect *)malloc(sizeof(vect));
	softmax_vect->v_size=a->v_size;
	softmax_vect->v=(floater *)malloc(a->v_size*sizeof(vect));
	for(i=0;i<softmax_vect->v_size;i++)
	{
		softmax_vect->v[i]=softmax(a, i);
	}
	
	return softmax_vect;
}



//cross entropy function
//  first argument is the target t, secomd argument is the predicted value y
floater cross_entropy(vect *t, vect *y)
{
	
	//first check whether the 2 vectors are of same size
	
	if(t->v_size!=y->v_size)
	{
		cout<<"Vectors in cross entropy vary in zsize"<<t->v_size<<" "<<y->v_size<<"\n" ;
		exit(0);
	}
	
	int i;
	float f1, f2;
	f1=0;
	
	for(i=0;i<t->v_size;i++)
	{
		f1=f1+(-1*t->v[i]*log(y->v[i]));
	}
	
	return f1;
	
	
}


//derivative of sigmoid function at x
floater derivative_sigmoid(floater x)
{
	floater f1;
	f1=sigmoid(x)*(1-sigmoid(x));
	return f1;
	
}


//return  the derivative of tanh at x
floater derivative_tanh(floater x)
{
	floater f1;
	f1=1-(tanh(x)*tanh(x));
	return f1;
	
}



//derivative of softmax 
floater derivative_softmax(vect *a, int index, int derivative_index)
{
	int  i,j;
	float f1, f2, f3;
	
	//two cases 
	if(index==derivative_index)
	{
		f1=softmax(a, index);
		f3=f1*(1-f1);
		
	}
	else
	{
		f1=softmax(a, index);
		f2=softmax(a, derivative_index);
		f3=-1*f1*f2;
	}
	
	return f3;
	
}
