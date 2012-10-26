#include <Node/Node.h>
#include "NodeTestMessages.pb.h"

#include <stdio.h>

int main()
{
    rpg::node n("Node2");
    n.set_verbocity( 3 ); // make some noise on errors

    // subscribe to Node1's topic
    if( n.subscribe( "Node1/Node1Topic" ) == false ) {
        printf("Error subscribing to topic.\n");
    }

    if( n.advertise( "Node2Topic" ) == false ) {
        printf("Error subscribing to topic.\n");
    }

    unsigned int nCount = 0;

    // test the "easy" api -- call Node1->SimpleRpcMethod("test")
    int res;
    n.call_rpc( "Node1/SimpleRpcMethod", "test", res );
    printf("Got %d back from 'Node1/SimpleRpcMethod'\n", res);

    // now make a mistake
    n.call_rpc( "Node1/SimpleRpcMethod2", "test", res );

    while(1) {
        Msg mMsg;
        n.receive( "Node1/Node1Topic", mMsg ); // blocking call
        printf("Got '%s'.\n", mMsg.value().c_str());

        nCount++;
        if( nCount == 3 ) {
            printf("--- Sending RPC message! ---\n");
            mMsg.set_value( "Bye!" );
            n.call_rpc( "Node1/RpcMethod", mMsg, mMsg );
        }

        sleep(1);
    }

    return 0;
}
