# cudaflow vs thrust

|                      	|           	|          	|
|----------------------	|-----------	|----------	|
| Type                 	| d         	|          	|
| #Elements            	| 269484032 	|          	|
| #Executions          	| 5         	|          	|
|                      	|           	|          	|
| Algorithm            	| Thrust    	| cudaFlow 	|
| find_if              	| 0.0187    	| 0.0064   	|
| transform            	| 0.0132    	| 0.0131   	|
| reduce               	| 0.00649   	| 0.00754  	|
| transform_reduce     	| 0.00657   	| 0.00637  	|
| transform_and_reduce 	| 0.015     	| 0.0149   	|
| sort                 	| 0.301     	| 0.288    	|
