using System;
using System.Collections.Generic;
using System.Linq;

using Nanon.Math.Linear;

namespace Nanon.Data
{
	public class DataSet<InputT, OutputT> : IDataSet<InputT, OutputT>
	{
		InputT[]  inputs;
		OutputT[] outputs;
		
		public DataSet(InputT[] inputsA, OutputT[] outputsA)
		{						
		   	inputs = inputsA;
			outputs = outputsA;
		}
		
		public static DataSet<Matrix, Matrix> FromFile(string inputsFileName, string outputsFileName) 
		{
			var inputs  = Matrix.FromFile(inputsFileName);
			var outputs = Matrix.FromFile(outputsFileName);
			return new DataSet<Matrix, Matrix>(inputs, outputs);
		}
		
		public DataSet<InputT, OutputT> Take(int count)
		{
			var newInputs = new InputT[count];
			Array.Copy(inputs, newInputs, count);
			
			var newOutputs = new OutputT[count];
			Array.Copy(outputs, newOutputs, count);
			
			return new DataSet<InputT, OutputT>(newInputs, newOutputs);			
		}
		
		public DataSet<InputT, OutputT> Take(int fromI, int toI)
		{
			var count = toI - fromI;
			var newInputs = new InputT[count];
			Array.Copy(inputs, fromI, newInputs, 0, count);
			
			var newOutputs = new OutputT[count];
			Array.Copy(outputs, fromI, newOutputs, 0, count);
			
			return new DataSet<InputT, OutputT>(newInputs, newOutputs);			
		}
		
		public DataSet<A, B> Convert<A, B>(Func<InputT, A> convInputs, Func<OutputT, B> convOutputs)
		{
		    return new DataSet<A, B>(
				Array.ConvertAll<InputT, A>(inputs,  x => convInputs(x)),
				Array.ConvertAll<OutputT, B>(outputs, x => convOutputs(x))
			);
		}
		
		public void TransformInputs(Action<InputT> convInputs)
		{
			foreach(var input in inputs)
				convInputs(input);
		}
		
		public IEnumerable<InputT> Inputs 
		{
			get 
			{
				return this.inputs;
			}
		}

		public IEnumerable<OutputT> Outputs 
		{
			get 
			{
				return this.outputs;
			}
		}
		
		public IEnumerable<Tuple<InputT, OutputT> > Set 
		{ 
			get
			{
				return inputs.Zip(outputs, (a, b) => Tuple.Create(a, b));
			}
		}
		
		public int Size 
		{ 
			get
			{
				return inputs.Length;
			}
		}	
		
		public OutputT FirstOutput
		{
			get 
			{
				return outputs[0];	
			}
		}
		
		public InputT FirstInput
		{
			get
			{
				return inputs[0];	
			}
		}
	}
}

