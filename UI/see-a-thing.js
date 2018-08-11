const SOCKET_URL = "ws://0.0.0.0:5000/";

InitializeSocketComms = () => {

    var socket = new WebSocket(SOCKET_URL)

    socket.onopen = () => {
      console.log("Socket Open")
    }

    socket.onmessage = (event) => {
      console.log(event.data)

      return false
    }

    socket.onclose = (event) => {
      console.log("Retrying Connection in a few seconds.. ")
      setTimeout(() => InitializeSocketComms(), 10000);
      return false
    }
}

window.onload = InitializeSocketComms
