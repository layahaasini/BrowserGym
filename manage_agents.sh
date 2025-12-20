#!/bin/bash
# Manage BrowserGym Agents
# Usage: ./manage_agents.sh [start|stop|restart|status]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

case "${1:-help}" in
    start)
        echo "Starting both agents..."
        
        # Start green agent in background
        echo "Starting green agent on port 8000..."
        ./run_green.sh &
        GREEN_PID=$!
        echo "Green agent started (PID: $GREEN_PID)"
        
        sleep 2
        
        # Start white agent in background
        echo "Starting white agent on port 8001..."
        ./run_white.sh &
        WHITE_PID=$!
        echo "White agent started (PID: $WHITE_PID)"
        
        echo ""
        echo "✅ Both agents started!"
        echo "Green agent: http://localhost:8000"
        echo "White agent: http://localhost:8001"
        ;;
        
    stop)
        echo "Stopping agents..."
        pkill -f "green_agent.py"
        pkill -f "white_agent.py"
        echo "✅ Agents stopped"
        ;;
        
    restart)
        echo "Restarting agents..."
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo "Checking agent status..."
        echo ""
        echo "Green agent processes:"
        ps aux | grep "[g]reen_agent.py" || echo "  Not running"
        echo ""
        echo "White agent processes:"
        ps aux | grep "[w]hite_agent.py" || echo "  Not running"
        echo ""
        echo "Testing endpoints:"
        echo -n "  Green agent (8000): "
        curl -s http://localhost:8000/status 2>/dev/null && echo "✅ OK" || echo "❌ Not responding"
        echo -n "  White agent (8001): "
        curl -s http://localhost:8001/status 2>/dev/null && echo "✅ OK" || echo "❌ Not responding"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start both green and white agents"
        echo "  stop    - Stop both agents"
        echo "  restart - Restart both agents"
        echo "  status  - Check if agents are running"
        exit 1
        ;;
esac

