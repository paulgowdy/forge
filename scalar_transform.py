from transformer import TransformerACT, TransformerBlock
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input,Lambda, Subtract, Concatenate
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


'''
want to adapt the transformer torso + scalar torso approach
a simpler version of AlphaStar's basic architecture

so I have the basic transformer working here
takes a sequence of 1-Dim arrays (boards)
applies self attention to them

the output is the same shape as the input (so a sequence of boards)
that will basically be "highlighted" for importance...

one idea:
apply some basic Conv2Ds to this stack
flatten
then combine with 'dense-processed' scalar inputs (step, each player's halite, possibly # ships and shipyards as well)

HMM - potential big problem
that will work for just the "halite board", the record of all the halite
but I actually want a stack for each board - halite board, ship board, yard board
I'm simplifying a lot of this away in my single player set up
but its gonna get very big very fast

another idea:
so think about the transformer
its sequence to sequence
the sequences im dealing with are short time stacks of boards
maybe the sequence I should try to predict directly is the sequence of orders
there's not really a known 'right seq', not really a decoder - right?

IMAGE TRANSFORMER - that's another approach
honestly might make more sense to treat the board as, you know, a board
21x21 board, thats 400+ length sequence
pretty long - could be a problem with sheer computational size...

im gonna need to find a better way to get compute
think it might be G... check on the rules first
if I can use G, then start this weekend
ALSO WRITE PAPER!!!

might be able to 'encode' the board in some way
possibly even just max pooling or something


(also need to be prepared to adapt this to multiple ships/yards)
I think my fixed length output vector
with the ship/yard select array inputs as well
is still the best approach


'''

#tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()

BOARD_SIZE = 5
SEQUENCE_LENGTH = 4
BATCH_SIZE = 2
ACTION_SPACE = 5
SCALAR_SIZE = 2

flat_board_len = BOARD_SIZE ** 2
# The input will be:
# A batch of consecutive boards (SEQUENCE_LENGTH Boards),each flattened

# hmm, seems like I might need to try without batches...
#input_shape = (BATCH_SIZE, SEQUENCE_LENGTH, flat_board_len)
input_shape = (SEQUENCE_LENGTH, flat_board_len)

def halite_transformer_model(input_shape,
                scalar_input_len,
                action_space,
                transformer_depth, num_heads,
                transformer_dropout = 0.02,
                l2_reg_penalty = 1e-6,
                confidence_penalty_weight = 0.1):

                transformer_act_layer = TransformerACT(name='adaptive_computation_time')
                transformer_block = TransformerBlock(
                    name='transformer', num_heads=num_heads,
                    residual_dropout=transformer_dropout,
                    attention_dropout=transformer_dropout,
                    use_masking=True, vanilla_wiring=False)

                scalar_inputs = Input(shape=(scalar_input_len,))
                scalar_dense1 = Dense(16)(scalar_inputs)
                scalar_dense2 = Dense(8)(scalar_dense1)

                board_sequence_input = Input(shape=input_shape, dtype='float64')
                next_step_input = board_sequence_input

                for i in range(transformer_depth):
                    #next_step_input = coordinate_embedding_layer(next_step_input, step=i)
                    next_step_input = transformer_block(next_step_input)
                    next_step_input, act_output = transformer_act_layer(next_step_input)
                    # which one of these two do I actually want...

                transformer_act_layer.finalize()
                #next_step_input = act_output
                reshaped = tf.reshape(act_output, (input_shape[0], int(np.sqrt(input_shape[1])), int(np.sqrt(input_shape[1]))))
                reshaped = tf.transpose(reshaped, [1,2,0])
                reshaped = tf.expand_dims(reshaped, 0)

                conv1 = Conv2D(16, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(reshaped)
                conv2 = Conv2D(16, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(conv1)
                flat = Flatten()(conv2)

                combined = Concatenate(axis=1)([flat, scalar_dense2])

                dense1 = Dense(16)(combined)
                dense2 = Dense(8)(dense1)

                # ultimately want to combine the transformer sequence input
                # with the static (i.e. one frome) scalar values and item select inputs

                #word_predictions = output_softmax_layer(output_layer([next_step_input, embedding_matrix]))
                order_predictions = Dense(action_space)(dense2)

                model = Model(inputs=[board_sequence_input, scalar_inputs], outputs=[order_predictions])
                # Penalty for confidence of the output distribution, as described in
                # "Regularizing Neural Networks by Penalizing Confident
                # Output Distributions" (https://arxiv.org/abs/1701.06548)
                confidence_penalty = K.mean(
                    confidence_penalty_weight *
                    K.sum(order_predictions * K.log(order_predictions), axis=-1))

                model.add_loss(confidence_penalty)
                return model

model = halite_transformer_model(input_shape,
                SCALAR_SIZE,
                ACTION_SPACE,
                4, 5,
                transformer_dropout = 0.02,
                l2_reg_penalty = 1e-6,
                confidence_penalty_weight = 0.1)

model.summary()
# need to compile before training!

board_input = np.array([np.random.random(input_shape)])
scalar_input = np.array([[0,2]])
print('scalar input shape', scalar_input.shape)

z = model.predict([board_input, scalar_input])

print(z)
print(z.shape)
