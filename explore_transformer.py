from transformer import TransformerACT, TransformerBlock
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

#tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()

BOARD_SIZE = 5
SEQUENCE_LENGTH = 4
BATCH_SIZE = 2

flat_board_len = BOARD_SIZE ** 2
# The input will be:
# A batch of consecutive boards (SEQUENCE_LENGTH Boards),each flattened

# hmm, seems like I might need to try without batches...
#input_shape = (BATCH_SIZE, SEQUENCE_LENGTH, flat_board_len)
input_shape = (SEQUENCE_LENGTH, flat_board_len)

def halite_transformer_model(input_shape,
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


                board_sequence_input = Input(shape=input_shape, dtype='float64')
                next_step_input = board_sequence_input

                for i in range(transformer_depth):
                    #next_step_input = coordinate_embedding_layer(next_step_input, step=i)
                    next_step_input = transformer_block(next_step_input)
                    next_step_input, act_output = transformer_act_layer(next_step_input)

                transformer_act_layer.finalize()
                next_step_input = act_output

                # ultimately want to combine the transformer sequence input
                # with the static (i.e. one frome) scalar values and item select inputs

                #word_predictions = output_softmax_layer(output_layer([next_step_input, embedding_matrix]))
                order_predictions = next_step_input

                model = Model(inputs=[board_sequence_input], outputs=[order_predictions])
                # Penalty for confidence of the output distribution, as described in
                # "Regularizing Neural Networks by Penalizing Confident
                # Output Distributions" (https://arxiv.org/abs/1701.06548)
                confidence_penalty = K.mean(
                    confidence_penalty_weight *
                    K.sum(order_predictions * K.log(order_predictions), axis=-1))

                model.add_loss(confidence_penalty)
                return model

model = halite_transformer_model(input_shape,
                4, 5,
                transformer_dropout = 0.02,
                l2_reg_penalty = 1e-6,
                confidence_penalty_weight = 0.1)

model.summary()

input = np.array([np.random.random(input_shape)])
z = model.predict(input)
print(z)
print(z.shape)
