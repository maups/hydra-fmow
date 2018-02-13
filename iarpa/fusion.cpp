#include <bits/stdc++.h>

using namespace std;

map<int,vector<string> > labels;

int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "ERROR: Need at least one file to do predictions!" << endl;
		return 1;
	}

	int id;
	string s, label;
	for(int i=1; i < argc; i++) {
		ifstream fp(argv[i], ifstream::in);
		while(fp >> s) {
			size_t pos;
			id = stoi(s, &pos);
			s = s.substr(pos+1);
			label = s.substr(0,s.find(";"));
			labels[id].push_back(label);
		}
		fp.close();
	}

	int thr = (argc-1)/2;

	for(auto it = labels.begin(); it != labels.end(); it++) {
		sort(it->second.begin(), it->second.end());
		int first = 0;
		label = "false_detection";
		for(int i=1; i < it->second.size(); i++) {
			if(it->second[i] == it->second[first])
				continue;
			if(i-first > thr)
				label = it->second[first];
			first = i;
		}
		if(it->second.size()-first > thr)
			label = it->second[first];
		cout << it->first << "," << label << endl;
	}
}

