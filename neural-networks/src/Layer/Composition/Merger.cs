using System;

using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class Merger<InputT> : ISingleLayer<InputT[], Vector>
	{
		ISingleLayer<InputT, Matrix>[] layers;
		
		int size;       // layer count
		int outputSize; // size of output of each layer
		int outputWidth; //
		int outputHeight;
		
		Vector outputs;
		Vector signals;
		InputT[] predErrors;
		
		public Merger (ISingleLayer<InputT, Matrix>[] parLayers)
		{
			size = parLayers.Length;
			
			if (size == 0) 
				throw new ArgumentException("layer count should be more than zero");
			
			layers = parLayers;
			outputSize = parLayers[0].Output.Size;
			
			predErrors = new InputT[size];
			
			outputs = new Vector(size * outputSize);
			signals = new Vector(size * outputSize);
		}

		#region ISingleLayer[InputT[],Vector] implementation
		public Vector FeedForward (InputT[] input)
		{
			for (var i = 0; i < size; ++i)
			{
				var iOut = layers[i].FeedForward(input[i]);
				var iSig = layers[i].Signal;
				
				var iFrom = i * outputSize;
				var iTo   = iFrom + outputSize;
				
				outputs.Pack(iFrom, iTo, iOut.ToVector);
				signals.Pack (iFrom, iTo, iSig.ToVector);
			}
			
			return outputs;	
		}

		public InputT[] PropagateBackward (InputT[] input, InputT[] predSignal, Vector error)
		{
			for (var i = 0; i < size; ++i)
			{
				var iFrom = i * outputSize;
				var iTo   = iFrom + outputSize;
				var unwindedError = error.Cut(iFrom, iTo);
				var windedError = new Matrix(outputWidth, outputHeight, unwindedError.Cells);
				
				predErrors[i] = layers[i].PropagateBackward(input[i], predSignal[i], windedError);
			}
			return predErrors;
		}

		public void Gradient(InputT[] input, Vector outputError)
		{
			for (var i = 0; i < size; ++i)
			{
				// cut 
				var iFrom = i * outputSize;
				var iTo   = iFrom + outputSize;
				var unwindedError = outputError.Cut(iFrom, iTo);

				var windedError = new Matrix(outputWidth, outputHeight, unwindedError.Cells);
				layers[i].Gradient(input[i], windedError);
			}
		}

		public void Correct(double coeff)
		{
			for (var i = 0; i < size; ++i)
				layers[i].Correct(coeff);
		}

		public Vector Signal {
			get {
				return signals;
			}
		}

		public Vector Output {
			get {
				return outputs;
			}
		}
		#endregion
	}
}

