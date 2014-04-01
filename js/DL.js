Array.prototype.compare = function (array) {
    // if the other array is a falsy value, return
    if (!array)
        return false;

    // compare lengths - can save a lot of time
    if (this.length != array.length)
        return false;

    for (var i = 0, l=this.length; i < l; i++) {
        // Check if we have nested arrays
        if (this[i] instanceof Array && array[i] instanceof Array) {
            // recurse into the nested arrays
            if (!this[i].compare(array[i]))
                return false;
        }
        else if (this[i] != array[i]) {
            // Warning - two different object instances will never be equal: {x:20} != {x:20}
            return false;
        }
    }
    return true;
}

function DeepLearning(gameManager) {
    this.gameManager = gameManager;
    this.lastScore = this.gameManager.score;
    this.lastGameGrid = this.convertGrid(this.gameManager.grid);

    var num_inputs = 16;
    var num_actions = 4;
    var temporal_window = 1; // amount of temporal memory. 0 = agent lives in-the-moment :)
    var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

    // the value function network computes a value of taking any of the possible actions
    // given an input state. Here we specify one explicitly the hard way
    // but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
    // to just insert simple relu hidden layers.
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
    layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
    layer_defs.push({type:'regression', num_neurons:num_actions});

    // options for the Temporal Difference learner that trains the above net
    // by backpropping the temporal difference learning rule.
    var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

    var opt = {};
    opt.temporal_window = temporal_window;
    opt.experience_size = 30000;
    opt.start_learn_threshold = 1000;
    opt.gamma = 0.7;
    opt.learning_steps_total = 200000;
    opt.learning_steps_burnin = 3000;
    opt.epsilon_min = 0.05;
    opt.epsilon_test_time = 0.05;
    opt.layer_defs = layer_defs;
    opt.tdtrainer_options = tdtrainer_options;

    this.brain = new deepqlearn.Brain(num_inputs, num_actions, opt); // woohoo

    document.getElementsByClassName('restart-button')[0].onclick = this.start();
}

DeepLearning.prototype = {
    simulateKeyPress: function (keyCode) {
//      var keyboardEvent = document.createEvent("KeyboardEvent");
//      var initMethod = typeof keyboardEvent.initKeyboardEvent !== 'undefined' ? "initKeyboardEvent" : "initKeyEvent";
//
//      keyboardEvent[initMethod](
//                          "keypress", // event type : keydown, keyup, keypress, "keydown"
//                          true, // bubbles
//                          true, // cancelable
//                          window, // viewArg: should be window
//                          false, // ctrlKeyArg
//                          false, // altKeyArg
//                          false, // shiftKeyArg
//                          false, // metaKeyArg
//                          keyCode, // keyCodeArg : unsigned long the virtual key code, else 0
//                          0 // charCodeArgs : unsigned long the Unicode character associated with the depressed key, else 0
//      );
//      document.dispatchEvent(keyboardEvent);

        var eventObj = document.createEventObject ?
            document.createEventObject() : document.createEvent("Events");

        if(eventObj.initEvent){
          eventObj.initEvent("keydown", true, true);
        }

        eventObj.keyCode = keyCode;
        eventObj.which = keyCode;

        document.dispatchEvent ? document.dispatchEvent(eventObj) : document.fireEvent("onkeydown", eventObj);
    },

    convertGrid: function (grid) {
        var result = {
            newGrid: [],
            max: 0,
            emptyCount: 0
        }
        for (i=0; i<grid.size; i++) {
            result.newGrid[i] = [];
            for(j=0; j<grid.size; j++) {
                if (grid.cells[i][j]) {
                    result.newGrid[i][j] = grid.cells[i][j].value;
                    result.max = result.max < grid.cells[i][j].value ? grid.cells[i][j].value : result.max;
                }
                else {
                    result.newGrid[i][j] = 0;
                    result.emptyCount++;
                }

            }
        }
        return result;
    },

    makeAction: function () {
        this.currentGameGrid = this.convertGrid(this.gameManager.grid);
        var actionId = this.brain.forward(this.currentGameGrid.newGrid);
        switch (actionId) {
            case 0:
                this.simulateKeyPress(37);
                break;
            case 1:
                this.simulateKeyPress(38);
                break;
            case 2:
                this.simulateKeyPress(39);
                break;
            case 3:
                this.simulateKeyPress(40);
        }
        this.backward(this.currentGameGrid);
    },

    backward: function (lastGameGrid) {
        var reward = 0;

        var scoreDelta = this.gameManager.score - this.lastScore;
        this.currentGameGrid = this.convertGrid(this.gameManager.grid);
        this.lastScore = this.gameManager.score;

        // no change was done? bad reward
        if (lastGameGrid.newGrid.compare(this.currentGameGrid.newGrid)) {
            console.log('reward: -1');
            return this.brain.backward(-1);
        }

        // awarding for more empty cells
        //if (this.currentGameGrid.emptyCount > lastGameGrid.emptyCount) {
        //    reward += (16 / (lastGameGrid.emptyCount - this.currentGameGrid.emptyCount));
        //}

        reward += scoreDelta / 1000;

        if (this.gameManager.won) {
            reward += 10;
        }

        if (this.gameManager.over) {
            setTimeout(function() {
                document.getElementsByClassName('retry-button')[0].click();
            }, 1500);
            reward = -10;
        }

        console.log('reward: ' + reward);
        this.brain.backward(reward);
    },

    start: function() {
        this.interval = setInterval(this.makeAction.bind(this), 300);
    }
}
