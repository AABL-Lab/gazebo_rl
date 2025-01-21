# accept the start and end idxs from the command line
# and replay the episodes from start to end

# list of integers 
trials=(335 333 332 330 329 328 327 326 325 321 320 319 318 317 316 311 309 308 306 305 304 303 302 301 299 298 297 290 286 284 283 281 280 279 278 277 276 275 274 273 269 267 266 265 263 262 261 259 258 257 256 255 252 251 250 249 248 247 246 245 244 243 242 239 237 236 235 234 233 232)

# if -r is an argument, pick a random trial
if [ "$1" == "-r" ]; then
    trials=($(shuf -e "${trials[@]}"))
    # pick one trial
    trials=(${trials[0]})
fi

for i in "${trials[@]}"
do
    # keep the leading 0
    i=$(printf "%03d" $i)
    echo "Replaying episode $i"
    python publish_saved_video.py -d user_$i --dont-replay -na -v -c 700 -clo 200

    # if we've recieved a ctrl-c, then exit
    if [ $? -ne 0 ]; then
        exit
    fi

done
